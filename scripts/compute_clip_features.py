"""Compute the CLIP image embeddings used to fit BrainACTIV modulation embeddings.

Writes <root>/subjXX/clip_features.npy, (n_images, 1024) float32, with one row
per image of NSD(partition="all"), i.e. every image the subject saw at least
once. Uses the same CLIP model as the BrainACTIV pipeline. Run
download_nsd_fmri.py and download_nsd_images.py first.

Example:
python scripts/compute_clip_features.py --root ~/Documents/Datasets/NSD          # all subjects
python scripts/compute_clip_features.py 1 2 --root ~/Documents/Datasets/NSD      # subjects 1 and 2
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # type: ignore
from brainactiv.datasets import NSD
from brainactiv.methods import CLIP


@torch.no_grad()
def embed(
	clip: CLIP,
	image_dir: Path,
	coco_ids: list[int],
	batch_size: int,
) -> dict[int, np.ndarray]:
	out = {}
	for start in tqdm(range(0, len(coco_ids), batch_size), desc="    embedding"):
		batch = coco_ids[start:start + batch_size]
		images = [Image.open(image_dir / f"{c}.png").convert("RGB") for c in batch]
		pixels = clip.processor(images=images, return_tensors="pt")["pixel_values"]
		feats = clip.clip(pixels.to(clip.device)).image_embeds.float().cpu().numpy()
		out.update(zip(batch, feats))
	return out


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("subjects", nargs="*", type=int, default=list(range(1, 9)), help="default: 1-8")
	p.add_argument("--root", type=Path, default=Path("./data"))
	p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
	p.add_argument("--batch-size", type=int, default=32)
	args = p.parse_args()
	root = args.root.expanduser()

	# Subjects share images, so embed each image once
	subject_ids = {
		s: NSD(root=str(root), subject=s, partition="all").coco_ids
		for s in args.subjects
	}
	unique_ids = sorted({int(c) for ids in subject_ids.values() for c in ids})

	# Check every image opens before the long embedding run (reads headers only, a few seconds)
	unreadable = []
	for c in unique_ids:
		try:
			Image.open(root / "images" / f"{c}.png").close()
		except Exception as e:  # noqa: BLE001
			unreadable.append(f"{c}.png ({type(e).__name__}: {e})")
	if unreadable:
		raise SystemExit(f"{len(unreadable)} unreadable images:\n  " + "\n  ".join(unreadable))

	print(f">>> Embedding {len(unique_ids)} images")
	clip = CLIP(args.device)
	feats = embed(clip, root / "images", unique_ids, args.batch_size)

	for s, ids in subject_ids.items():
		path = root / f"subj{s:02d}" / "clip_features.npy"
		np.save(path, np.stack([feats[int(c)] for c in ids]).astype(np.float32))
		print(f"    subj{s:02d}: {len(ids)} rows -> {path}")
	print(">>> Done")


if __name__ == "__main__":
	main()
