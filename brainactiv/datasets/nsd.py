import json
import os
import pathlib
import re
import numpy as np
from PIL import Image
from torch.utils import data
from typing import Literal, Optional, Union


class NSD(data.Dataset):

	def __init__(
		self,
		root: str,
		subject: Literal[1, 2, 3, 4, 5, 6, 7, 8],
		partition: Literal["train", "test", "all"],
		hemisphere: Optional[Literal["lh", "rh", "both"]] = None,
		roi: Optional[Union[str, list[str]]] = None,
		tval_threshold: float = 5.0,
		zscore: Optional[Literal["session"]] = "session",
		return_trial_average: bool = False,
		subset: Optional[Literal[
			"wild_animals",
			"vehicles",
			"sports",
			"food",
			"birds",
			"furniture",
		]] = None,
		dtype: type = np.float32,
	):
		super().__init__()
		assert subject in [1, 2, 3, 4, 5, 6, 7, 8]
		assert partition in ["train", "test", "all"]

		self.root = pathlib.Path(root)
		self.subject = subject
		self.partition = partition
		self.return_trial_average = return_trial_average
		self.zscore = zscore
		self.dtype = dtype
		self.return_activations = roi is not None
		self.subj_dir = self.root / f"subj{subject:02d}"
		self.coco_ids = np.load(self.subj_dir / "coco_ids.npy")

		sess_path = self.subj_dir / "session_id.npy"
		self.session_id = np.load(sess_path) if sess_path.exists() else None
		if self.session_id is None:
			self.n_reps = None
			if zscore == "session":
				raise FileNotFoundError(
					f"{sess_path} missing; session-wise z-scoring needs it. "
					"Re-run nsd_surface.py, or pass zscore=None.",
				)
		else:
			self.n_reps = (self.session_id >= 0).sum(1)

		mask = (
			self.load_partition_mask() &
			self.load_subset_mask(subset) &
			((self.n_reps > 0) if self.n_reps is not None else True)
		)
		self.partition_mask = mask.copy()
		self.coco_ids = self.coco_ids[self.partition_mask]

		if self.return_activations:
			assert roi is not None and hemisphere is not None
			self.hemisphere = hemisphere
			if hemisphere == "both":
				self.activations = np.concatenate([
					self.load_activations(roi, "lh", tval_threshold),
					self.load_activations(roi, "rh", tval_threshold),
				], axis=-1)
			else:
				self.activations = self.load_activations(roi, hemisphere, tval_threshold)
			if return_trial_average:
				self.activations = np.nanmean(self.activations, axis=1)
		self.partition_mask = self.partition_mask[(
			(self.n_reps > 0) if self.n_reps is not None else True
		)]
		if self.n_reps is not None:
			self.n_reps = self.n_reps[mask]

	def __len__(self) -> int:
		return len(self.coco_ids)

	def __getitem__(self, idx: int) -> tuple:
		coco_id = self.coco_ids[idx]
		img = Image.open(self.root / "images" / f"{coco_id}.png").convert("RGB")
		if self.return_activations:
			return img, coco_id, self.activations[idx]
		return img, coco_id

	def get_image(self, idx) -> Image.Image:
		img = Image.open(self.root / "images" / f"{self.coco_ids[idx]}.png")
		return img.convert("RGB").resize((425, 425))

	def get_clip_features(self) -> np.ndarray:
		embeds = np.load(self.subj_dir / "clip_features.npy")
		return embeds[self.partition_mask]

	def available_rois(self, hemisphere: str = "lh") -> list[str]:
		pat = re.compile(rf"^{hemisphere}\.(.+)_mask\.npy$")
		return sorted(
			m.group(1) for m in
			(pat.match(f.name) for f in (self.subj_dir / "roi").glob("*_mask.npy"))
			if m
		)

	def _resolve_roi(self, name: str, hemisphere: str) -> str:
		if (self.subj_dir / "roi" / f"{hemisphere}.{name}_mask.npy").exists():
			return name
		cands = [r for r in self.available_rois(hemisphere) if r.split(".")[-1] == name]
		if len(cands) == 1:
			return cands[0]
		if len(cands) > 1:
			raise ValueError(
				f"ROI '{name}' is ambiguous -- it exists in several groups: "
				f"{cands}. Pass one of those qualified names.",
			)
		raise FileNotFoundError(
			f"No ROI '{name}' for {hemisphere}. "
			f"Call .available_rois('{hemisphere}') to list them.",
		)

	def _roi_mask(
		self,
		roi,
		hemisphere: str,
		tval_threshold: float,
		n_vertices: int,
	) -> np.ndarray:
		if isinstance(roi, str) and roi == "whole_brain":
			return np.ones(n_vertices, dtype=bool)
		names = [roi] if isinstance(roi, str) else list(roi)
		out = np.zeros(n_vertices, dtype=bool)
		for name in names:
			resolved = self._resolve_roi(name, hemisphere)
			m = np.load(self.subj_dir / "roi" / f"{hemisphere}.{resolved}_mask.npy")
			# Threshold per ROI by its own t-map, where one exists. Deciding by
			# file existence rather than a hardcoded list means newly included
			# groups behave correctly.
			tpath = self.subj_dir / "roi" / f"{hemisphere}.{resolved}_tval.npy"
			if tpath.exists():
				m = m & (np.load(tpath) > tval_threshold)
			out |= m
		return out

	def load_activations(
		self,
		roi: Union[str, list[str]],
		hemisphere: Literal["lh", "rh"],
		tval_threshold: float,
	) -> np.ndarray:

		path = self.subj_dir / f"{hemisphere}.fmri_data.npy"
		mm = np.load(path, mmap_mode="r")
		if mm.ndim != 3:
			raise ValueError(
				f"{path} has shape {mm.shape}; expected (n_images, 3, n_vertices). "
				"This class needs the per-repetition data.",
			)
		n_vertices = mm.shape[2]
		roi_mask = self._roi_mask(roi, hemisphere, tval_threshold, n_vertices)
		if not roi_mask.any():
			raise ValueError(
				f"ROI selection is empty for {hemisphere} "
				f"(tval_threshold={tval_threshold} may be too strict).",
			)

		act: np.ndarray = np.asarray(mm[:, :, roi_mask], dtype=self.dtype)
		if self.zscore == "session":
			act = self._zscore_by_session(act)
		act = act[self.partition_mask]
		return act

	def _zscore_by_session(self, act: np.ndarray) -> np.ndarray:
		if self.session_id is None:
			return act
		for s in np.unique(self.session_id[self.session_id >= 0]):
			m = self.session_id == s
			blk = act[m]
			if blk.size == 0:
				continue
			sd = blk.std(0)
			act[m] = (blk - blk.mean(0)) / np.where(sd == 0, 1.0, sd)
		return act

	def average_repetitions(self, activations: Optional[np.ndarray] = None):
		a = self.activations if activations is None else activations
		if a.ndim < 3:
			return a
		return np.nanmean(a, axis=1)

	def load_partition_mask(self) -> np.ndarray:
		shared1000 = np.load(self.root / "shared1000.npy")
		mask = np.isin(self.coco_ids, shared1000)
		if self.partition == "train":
			mask = ~mask
		elif self.partition == "all":
			mask = np.ones_like(mask, dtype=bool)
		return mask

	def load_subset_mask(self, subset: Optional[str]) -> np.ndarray:
		if subset is None:
			return np.ones(self.coco_ids.shape, dtype=bool)
		subset_indices = set(get_subset_indices(self, subset).tolist())
		return np.array([int(i) in subset_indices for i in self.coco_ids], dtype=bool)


supercategories = {
	"vehicle": [
		"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	],
	"outdoor": [
		"traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
	],
	"animal": [
		"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	],
	"wild_animal": [
		"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	],
	"accessory": [
		"hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
	],
	"sports": [
		"frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
		"skateboard", "surfboard", "tennis racket",
	],
	"kitchen": [
		"bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
	],
	"food": [
		"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
		"donut", "cake",
	],
	"furniture": [
		"chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
		"toilet", "door",
	],
	"electronic": [
		"tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	],
	"appliance": [
		"microwave", "oven", "toaster", "sink", "refrigerator", "blender",
	],
	"indoor": [
		"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush",
	],
}

subsets = {
	"wild_animals": {
		"all_positives": [["wild_animal"]],
		"negatives": ["person", "vehicle", "food"],
	},
	"vehicles": {
		"all_positives": [["vehicle"]],
		"negatives": ["person", "animal", "food"],
	},
	"sports": {
		"all_positives": [["person", "sports"]],
		"negatives": ["animal", "vehicle", "food"],
	},
	"food": {
		"all_positives": [["food"]],
		"negatives": ["person", "animal", "vehicle"],
	},
	"birds": {
		"all_positives": [["bird"]],
		"negatives": ["person", "food", "vehicle", "wild_animal", "cat", "dog"],
	},
	"furniture": {
		"all_positives": [["furniture"]],
		"negatives": ["person", "food", "vehicle", "animal"],
	},
}

SUBSETS = list(subsets.keys())

def get_subset_indices(nsd, subset):
	assert subset in SUBSETS
	all_positives = subsets[subset]["all_positives"]
	negatives = subsets[subset]["negatives"]
	all_indices = set()
	with open(os.path.join(nsd.root, "category2coco_ids.json")) as f:
		category2coco_ids = json.load(f)
	with open(os.path.join(nsd.root, "coco_id2categories.json")) as f:
		coco_id2categories = json.load(f)
	for cat, elements in supercategories.items():
		idxs = [category2coco_ids.get(el, []) for el in elements]
		idxs = np.unique(np.concatenate(idxs)).astype(int).tolist() if idxs else []
		category2coco_ids[cat] = idxs
		for idx in idxs:
			coco_id2categories[str(idx)].append(cat)
	for positives in all_positives:
		shared = set(category2coco_ids[positives[0]])
		for positive in positives:
			shared = set.intersection(shared, set(category2coco_ids[positive]))
		shared = shared.intersection(set(nsd.coco_ids.tolist()))
		for idx in shared:
			categories = coco_id2categories[str(idx)]
			if not any(c in negatives for c in categories):
				all_indices.add(idx)
	return np.array(sorted(all_indices))
