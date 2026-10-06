"""Download and preprocess the NSD surface data used by the BrainACTIV experiments.

Writes, for each subject, to <out>/subjXX/:
  coco_ids.npy            (10000,) COCO id of each of the subject's images
  session_id.npy          (10000, 3) scan session of each repetition, -1 if missing
  {lh,rh}.fmri_data.npy   (10000, 3, n_vertices) float16 betas, NaN if missing
  roi/{hemi}.{roi}_mask.npy and, for category-selective ROIs, roi/{hemi}.{roi}_tval.npy
and <out>/shared1000.npy, the COCO ids of the 1000 images seen by every subject.

Only vertices inside the ROI_GROUPS parcellations are kept.
Each subject needs ~30 GB of temporary download; session files are deleted once processed.

Example:
python scripts/download_nsd_fmri.py --out ~/Documents/Datasets/NSD          # all subjects
python scripts/download_nsd_fmri.py 1 2 --out ~/Documents/Datasets/NSD      # subjects 1 and 2
"""

import argparse
import csv
import sys
import urllib.request
import h5py  # type: ignore
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.io import loadmat  # type: ignore
from tqdm import tqdm  # type: ignore


BASE = "https://natural-scenes-dataset.s3.amazonaws.com"
BETA_VERSION = "betas_fithrf_GLMdenoise_RR"
INT16_SCALE = 300.0
N_IMAGES = 10000
MAX_REPS = 3
HEMIS = ("lh", "rh")
N_SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}

# ROI parcellations to export. Their union also defines which vertices are kept.
ROI_GROUPS = (
	"prf-visualrois",
	"prf-eccrois",
	"floc-bodies",
	"floc-faces",
	"floc-places",
	"floc-words",
	"streams",
)

# Localizer t-maps for the category-selective groups. NSD drops the hyphen and
# pluralisation in these filenames (lh.floc-bodies.mgz -> lh.flocbodiestval.mgz).
TVAL_FOR_GROUP = {
	"floc-bodies": "flocbodiestval",
	"floc-faces": "flocfacestval",
	"floc-places": "flocplacestval",
	"floc-words": "floccharacterstval",
}


def download(key: str, dest: Path, base: str = BASE) -> Path:
	"""Fetch `base/key` to `dest`, resuming partial downloads and retrying on errors."""
	if dest.exists():
		return dest
	dest.parent.mkdir(parents=True, exist_ok=True)
	tmp = dest.with_name(dest.name + ".part")
	for attempt in range(5):
		start = tmp.stat().st_size if tmp.exists() else 0
		try:
			req = urllib.request.Request(f"{base}/{key}")
			if start:
				req.add_header("Range", f"bytes={start}-")
			with urllib.request.urlopen(req, timeout=120) as r:
				if not (start and r.status == 206):
					start = 0
				total = start + int(r.headers.get("Content-Length") or 0)
				bar = tqdm(
					total=total,
					initial=start,
					unit="B",
					unit_scale=True,
					desc=f"    {dest.name}",
					leave=False,
				)
				with open(tmp, "ab" if start else "wb") as f, bar:
					while chunk := r.read(1 << 20):
						f.write(chunk)
						bar.update(len(chunk))
			tmp.rename(dest)
			return dest
		except Exception as e:  # noqa: BLE001
			print(f"    retry {attempt + 1}/5 ({type(e).__name__}: {e})")
	sys.exit(f"Failed to download {key}")


def coco_lookup(path: Path) -> np.ndarray:
	"""Array indexed by 0-based nsdId giving the COCO id."""
	out = np.full(73000, -1, np.int64)
	for r in csv.DictReader(path.read_text().splitlines()):
		out[int(r["nsdId"])] = int(r["cocoId"])
	return out


def trial_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""(session, 0-based nsdId, repetition) of every trial, in beta order."""
	rows = list(csv.DictReader(path.read_text().splitlines(), delimiter="\t"))
	rows.sort(key=lambda r: (int(r["SESSION"]), int(r["RUN"]), int(r["TRIAL"])))
	sess = np.array([int(r["SESSION"]) for r in rows], np.int16)
	nsd = np.array([int(r["73KID"]) - 1 for r in rows], np.int32)
	rep = np.empty(len(rows), np.int8)
	seen: dict = {}
	for i, v in enumerate(nsd):
		rep[i] = seen[v] = seen.get(v, -1) + 1
	return sess, nsd, rep


def read_betas(path: Path) -> np.ndarray:
	"""(750 trials, n_vertices) int16 betas of one session file."""
	with h5py.File(path) as f:
		arr = np.asarray(f["betas"])
	return arr if arr.shape[0] == 750 else arr.T


def parse_ctab(text: str) -> dict[int, str]:
	"""FreeSurfer colour table -> {index: name}, skipping index 0 / 'Unknown'."""
	out = {}
	for line in text.splitlines():
		parts = line.split()
		if len(parts) >= 2 and parts[0].isdigit():
			idx, name = int(parts[0]), parts[1]
			if idx > 0 and name.lower() not in ("unknown", "none"):
				out[idx] = name
	return out


def load_surface(path: Path) -> np.ndarray:
	return np.asarray(nib.load(str(path)).dataobj).squeeze()


def process_subject(subject: int, out: Path, cache: Path, expdesign: dict, coco: np.ndarray):
	subj = f"subj{subject:02d}"
	outdir = out / subj
	(outdir / "roi").mkdir(parents=True, exist_ok=True)
	print(f">>> {subj}")

	# ---- images and trials -------------------------------------------------
	my_images = expdesign["subjectim"][subject - 1].astype(np.int64) - 1  # 0-based nsdIds
	np.save(outdir / "coco_ids.npy", coco[my_images])
	img_pos = {int(v): i for i, v in enumerate(my_images)}

	behav = download(f"nsddata/ppdata/{subj}/behav/responses.tsv", cache / subj / "responses.tsv")
	t_sess, t_nsd, t_rep = trial_table(behav)

	# Which session each (image, repetition) came from; needed for session-wise z-scoring
	session_id = np.full((N_IMAGES, MAX_REPS), -1, np.int16)
	for s, nsd, rep in zip(t_sess, t_nsd, t_rep):
		session_id[img_pos[int(nsd)], rep] = s
	np.save(outdir / "session_id.npy", session_id)

	for hemi in HEMIS:
		# ---- ROIs ----------------------------------------------------------
		label_dir = f"nsddata/freesurfer/{subj}/label"
		labels = {}
		for g in ROI_GROUPS:
			name = f"{hemi}.{g}.mgz"
			labels[g] = load_surface(download(f"{label_dir}/{name}", cache / subj / name))
		keep = np.flatnonzero(np.any([lab > 0 for lab in labels.values()], axis=0))
		print(f"    {hemi}: keeping {keep.size} vertices")

		for group, lab in labels.items():
			ctab = download(f"{label_dir}/{group}.mgz.ctab", cache / subj / f"{group}.mgz.ctab")
			tval = None
			if group in TVAL_FOR_GROUP:
				stem = f"{hemi}.{TVAL_FOR_GROUP[group]}.mgz"
				tval = load_surface(download(f"{label_dir}/{stem}", cache / subj / stem))
				tval = tval[keep].astype(np.float32)
			for idx, name in parse_ctab(ctab.read_text()).items():
				np.save(outdir / "roi" / f"{hemi}.{name}_mask.npy", (lab == idx)[keep])
				if tval is not None:
					np.save(outdir / "roi" / f"{hemi}.{name}_tval.npy", tval)

		# ---- betas ---------------------------------------------------------
		arr = np.lib.format.open_memmap(
			outdir / f"{hemi}.fmri_data.npy",
			mode="w+",
			dtype=np.float16,
			shape=(N_IMAGES, MAX_REPS, keep.size),
		)
		arr[:] = np.nan
		for s in range(1, N_SESSIONS[subject] + 1):
			print(f"    {hemi} session {s}/{N_SESSIONS[subject]}")
			name = f"{hemi}.betas_session{s:02d}.hdf5"
			local = download(
				f"nsddata_betas/ppdata/{subj}/nativesurface/{BETA_VERSION}/{name}",
				cache / subj / name,
			)
			block = read_betas(local)[:, keep].astype(np.float32) / INT16_SCALE
			m = t_sess == s
			for row, nsd, rep in zip(block, t_nsd[m], t_rep[m]):
				arr[img_pos[int(nsd)], rep] = row
			local.unlink()
		arr.flush()
		del arr


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("subjects", nargs="*", type=int, default=list(range(1, 9)), help="default: 1-8")
	p.add_argument("--out", type=Path, default=Path("./data"))
	p.add_argument("--cache", type=Path, default=Path("./nsd_cache"))
	args = p.parse_args()
	out = args.out.expanduser()

	expdesign = loadmat(str(download(
		"nsddata/experiments/nsd/nsd_expdesign.mat",
		args.cache / "nsd_expdesign.mat",
	)))
	coco = coco_lookup(download(
		"nsddata/experiments/nsd/nsd_stim_info_merged.csv",
		args.cache / "nsd_stim_info_merged.csv",
	))
	out.mkdir(parents=True, exist_ok=True)
	np.save(out / "shared1000.npy", coco[expdesign["sharedix"].ravel() - 1])

	for subject in args.subjects:
		process_subject(subject, out, args.cache, expdesign, coco)
	print(f">>> Done: {out}")


if __name__ == "__main__":
	main()
