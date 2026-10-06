"""
DINOv2 encoding model of NSD fMRI responses on the fsaverage surface: the Memory Encoding Model
of Yang, Gee & Shi (2023), a LoRA-finetuned DINOv2 ViT-B/14. From BERG
(https://github.com/gifale95/BERG, model "fmri-nsd_fsaverage-huze").
"""

import numpy as np
import torch
import berg.models.fmri.huze.point_pe as point_pe
from pathlib import Path
from typing import Literal, Union
from PIL import Image
from berg import BERG
from berg.models.fmri.nsd_fsaverage_huze import HUZE
from ..utils import resolve_device
from .berg_utils import download_berg_file, to_stimulus


MODEL_ID = "fmri-nsd_fsaverage-huze"
MODEL_DIR = "encoding_models/modality-fmri/train_dataset-nsd_fsaverage/model-huze"
N_VERTICES = 163842  # per fsaverage hemisphere
# BERG's model card misspells OWFA as "OVWFA"; the metadata uses OWFA
DINOV2_ROIS = ["OWFA" if r == "OVWFA" else r for r in HUZE.VALID_ROIS]


def _sinusoidal(positions, features=16, periods=10000):
	"""BERG's point_pe.sinusoidal, with the frequencies built on CPU (logspace has no MPS kernel)."""
	dtype = positions.dtype if positions.is_floating_point() else None
	omega = torch.logspace(0, 1 / features - 1, features, periods, dtype=dtype).to(positions.device)
	fraction = omega * positions.unsqueeze(-1)
	return torch.stack((fraction.sin(), fraction.cos()), dim=-1)


point_pe.sinusoidal = _sinusoidal


class DINOv2Encoder:
	"""Predicts a subject's z-scored fMRI responses on the fsaverage vertices of one or more ROIs.

	Vertices are in fsaverage order (the union of the ROIs), left hemisphere before right when
	hemisphere="both".
	"""

	def __init__(
		self,
		subject: int,
		roi: Union[str, list[str]],
		hemisphere: Literal["lh", "rh", "both"] = "both",
		berg_dir: Union[str, Path] = "checkpoints/berg",
		device=None,
	):
		if hemisphere not in ("lh", "rh", "both"):
			raise ValueError(f"hemisphere must be lh, rh or both, got {hemisphere!r}")
		rois = [roi] if isinstance(roi, str) else list(roi)
		unknown = [r for r in rois if r not in DINOV2_ROIS]
		if unknown:
			raise ValueError(f"No DINOv2 model for ROI(s) {unknown}; available: {DINOV2_ROIS}")
		self.subject = subject
		self.rois = rois
		self.hemisphere = hemisphere

		weights_dir = f"{MODEL_DIR}/encoding_models_weights"
		metadata_key = f"{MODEL_DIR}/metadata/metadata_subject-{subject:02d}.npy"
		metadata_path = download_berg_file(berg_dir, metadata_key)
		for part in ("part1", "part2"):
			download_berg_file(berg_dir, f"{weights_dir}/subj{subject:02d}_{part}.pth")
		download_berg_file(berg_dir, f"{weights_dir}/part1_voxel_indices.pt")

		# Select the ROI vertices with masks rather than BERG's "roi" key, which rejects OWFA
		# and only takes one ROI. Both hemispheres are computed and sliced in encode().
		rois_meta = np.load(metadata_path, allow_pickle=True).item()["fmri"]
		masks = {}
		for hemi in ("lh", "rh"):
			masks[hemi] = np.zeros(N_VERTICES, dtype=np.int64)
			for r in rois:
				masks[hemi][rois_meta[f"{hemi}_fsaverage_rois"][r]] = 1
		self.n_lh = int(masks["lh"].sum())
		self.berg = BERG(berg_dir=str(berg_dir))
		self.model = self.berg.get_encoding_model(
			MODEL_ID,
			device=str(resolve_device(device)),
			subject=subject,
			selection={"lh_vertices": masks["lh"], "rh_vertices": masks["rh"]},
		)

	def encode(self, images: Union[Image.Image, list[Image.Image]]) -> np.ndarray:
		"""(n_vertices,) for one image, (n_images, n_vertices) for a list of square images."""
		single = isinstance(images, Image.Image)
		images = [images] if single else images
		responses = []
		for img in images:
			lh, rh = self.berg.encode(self.model, to_stimulus(img), show_progress=False)
			parts = {"lh": [lh[0]], "rh": [rh[0]], "both": [lh[0], rh[0]]}[self.hemisphere]
			responses.append(np.concatenate(parts))
		responses = np.stack(responses)
		return responses[0] if single else responses
