"""
Feature-weighted receptive field (fwRF) encoding models of NSD fMRI responses, from BERG
(https://github.com/gifale95/BERG, model "fmri-nsd-fwrf"). fwRF: St-Yves & Naselaris (2018).
"""

import numpy as np
from pathlib import Path
from typing import Union
from PIL import Image
from berg import BERG
from berg.models.fmri.nsd_fwrf import FMRIEncodingModel
from ..utils import resolve_device
from .berg_utils import download_berg_file, to_stimulus


MODEL_ID = "fmri-nsd-fwrf"
WEIGHTS_DIR = "encoding_models/modality-fmri/train_dataset-nsd/model-fwrf/encoding_models_weights"
FWRF_ROIS = list(FMRIEncodingModel.VALID_ROIS)
SPLIT_ROIS = ["lateral", "ventral"]  # their voxels are split across two weight files


class FWRFEncoder:
	"""Predicts a subject's z-scored fMRI responses in one or more ROIs from images.

	Responses are in NSD volume space and cover both hemispheres. With several ROIs,
	their voxels are concatenated in the given order.
	"""

	def __init__(
		self,
		subject: int,
		roi: Union[str, list[str]],
		berg_dir: Union[str, Path] = "checkpoints/berg",
		device=None,
	):
		rois = [roi] if isinstance(roi, str) else list(roi)
		unknown = [r for r in rois if r not in FWRF_ROIS]
		if unknown:
			raise ValueError(f"No fwrf model for ROI(s) {unknown}; available: {FWRF_ROIS}")
		self.subject = subject
		self.rois = rois
		for r in rois:
			splits = [f"_split-{i}" for i in (1, 2)] if r in SPLIT_ROIS else [""]
			for split in splits:
				key = f"{WEIGHTS_DIR}/weights_sub-{subject:02d}_roi-{r}{split}.pt"
				download_berg_file(berg_dir, key)
		self.berg = BERG(berg_dir=str(berg_dir))
		device = str(resolve_device(device))
		self.models = [
			self.berg.get_encoding_model(
				MODEL_ID, device=device, subject=subject, selection={"roi": r},
			)
			for r in rois
		]

	def encode(self, images: Union[Image.Image, list[Image.Image]]) -> np.ndarray:
		"""(n_voxels,) for one image, (n_images, n_voxels) for a list of square images."""
		single = isinstance(images, Image.Image)
		images = [images] if single else images
		responses = np.stack([
			np.concatenate([self.berg.encode(model, to_stimulus(img))[0] for model in self.models])
			for img in images
		])
		return responses[0] if single else responses
