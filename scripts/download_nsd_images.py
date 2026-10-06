"""Download the 73k NSD stimulus images used by the BrainACTIV experiments.

Writes <out>/images/{coco_id}.png, 425x425 RGB, for every NSD image.

The images are rebuilt from the original COCO photos: cropped with NSD's
cropBox and resized with PIL bicubic. This needs the COCO train2017 and
val2017 zips (~20 GB), which are downloaded to the cache and deleted once every
image is written. Images that already exist are skipped, so the script can be
re-run after an interruption.

Example:
python scripts/download_nsd_images.py --out ~/Documents/Datasets/NSD
"""

import argparse
import ast
import csv
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # type: ignore
from download_nsd_fmri import download


COCO_BASE = "http://images.cocodataset.org"
SPLITS = ("train2017", "val2017")
SIZE = 425


def nsd_crop(img: Image.Image, crop_box: tuple) -> Image.Image:
	"""Crop a COCO photo to the NSD field of view and resize it to SIZE x SIZE."""
	# cropBox gives the fraction of the image cut from each side
	top, bottom, left, right = crop_box
	w, h = img.size
	box = (int(w * left), int(h * top), int(w * (1 - right)), int(h * (1 - bottom)))
	return img.crop(box).resize((SIZE, SIZE), Image.BICUBIC)


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("--out", type=Path, default=Path("./data"))
	p.add_argument("--cache", type=Path, default=Path("./nsd_cache"))
	args = p.parse_args()
	out = args.out.expanduser() / "images"
	out.mkdir(parents=True, exist_ok=True)

	stim_info = download(
		"nsddata/experiments/nsd/nsd_stim_info_merged.csv",
		args.cache / "nsd_stim_info_merged.csv",
	)
	rows = list(csv.DictReader(stim_info.read_text().splitlines()))

	for split in SPLITS:
		todo = [
			r for r in rows
			if r["cocoSplit"] == split and not (out / f"{r['cocoId']}.png").exists()
		]
		if not todo:
			continue
		print(f">>> {split}: {len(todo)} images to write")
		archive = download(f"zips/{split}.zip", args.cache / f"{split}.zip", base=COCO_BASE)
		with zipfile.ZipFile(archive) as z:
			for r in tqdm(todo, desc="    writing"):
				coco_id = int(r["cocoId"])
				with z.open(f"{split}/{coco_id:012d}.jpg") as f:
					img = nsd_crop(Image.open(f).convert("RGB"), ast.literal_eval(r["cropBox"]))
				# Write then rename, so an interruption never leaves a truncated image
				tmp = out / f"{coco_id}.tmp"
				img.info.pop("icc_profile", None)
				img.save(tmp, format="PNG")
				tmp.rename(out / f"{coco_id}.png")
		archive.unlink()
	print(f">>> Done: {out}")


if __name__ == "__main__":
	main()
