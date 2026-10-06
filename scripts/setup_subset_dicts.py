"""Build the COCO category lookups used to define NSD image subsets.

Writes to <out>/:
  coco_id2categories.json   {coco_id: [category, ...]} for every NSD image
  category2coco_ids.json    {category: [coco_id, ...]}

An image is labelled with a category when that category's instance masks,
inside the NSD crop, cover at least MIN_PERCENT of the original COCO image.
Where instances overlap, the one with the higher category id wins.

Example:
python scripts/setup_subset_dicts.py --out ~/Documents/Datasets/NSD
"""

import argparse
import ast
import csv
import json
import zipfile
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO  # type: ignore
from tqdm import tqdm  # type: ignore
from download_nsd_fmri import download


COCO_BASE = "http://images.cocodataset.org"
MIN_PERCENT = 0.5


def load_coco(annotations: Path, split: str) -> COCO:
	coco = COCO()
	with zipfile.ZipFile(annotations) as z:
		coco.dataset = json.loads(z.read(f"annotations/instances_{split}.json"))
	coco.createIndex()
	return coco


def image_categories(coco: COCO, coco_id: int, crop_box: tuple) -> list[str]:
	info = coco.loadImgs(coco_id)[0]
	h, w = info["height"], info["width"]
	label_map = np.zeros((h, w), dtype=np.uint8)
	for ann in coco.loadAnns(coco.getAnnIds(imgIds=coco_id)):
		label_map = np.maximum(label_map, coco.annToMask(ann) * ann["category_id"])

	# cropBox gives the fraction of the image cut from each side
	top, bottom, left, right = crop_box
	label_map = label_map[
		round(h * top):h - round(h * bottom),
		round(w * left):w - round(w * right),
	]
	cats, counts = np.unique(label_map, return_counts=True)
	return [
		coco.cats[int(c)]["name"]
		for c, n in zip(cats, counts)
		if c != 0 and 100 * n / (h * w) >= MIN_PERCENT
	]


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("--out", type=Path, default=Path("./data"))
	p.add_argument("--cache", type=Path, default=Path("./nsd_cache"))
	args = p.parse_args()
	out = args.out.expanduser()
	out.mkdir(parents=True, exist_ok=True)

	stim_info = download(
		"nsddata/experiments/nsd/nsd_stim_info_merged.csv",
		args.cache / "nsd_stim_info_merged.csv",
	)
	annotations = download(
		"annotations/annotations_trainval2017.zip",
		args.cache / "annotations_trainval2017.zip",
		base=COCO_BASE,
	)
	print(">>> Loading COCO annotations")
	coco = {split: load_coco(annotations, split) for split in ("train2017", "val2017")}

	coco_id2categories: dict[int, list[str]] = {}
	category2coco_ids: dict[str, list[int]] = {}
	rows = list(csv.DictReader(stim_info.read_text().splitlines()))
	for r in tqdm(rows, desc="    labelling"):
		coco_id = int(r["cocoId"])
		categories = image_categories(
			coco[r["cocoSplit"]],
			coco_id,
			ast.literal_eval(r["cropBox"]),
		)
		coco_id2categories[coco_id] = categories
		for c in categories:
			category2coco_ids.setdefault(c, []).append(coco_id)

	with open(out / "coco_id2categories.json", "w") as f:
		json.dump(coco_id2categories, f)
	with open(out / "category2coco_ids.json", "w") as f:
		json.dump(category2coco_ids, f)
	print(f">>> Done: {out}")


if __name__ == "__main__":
	main()
