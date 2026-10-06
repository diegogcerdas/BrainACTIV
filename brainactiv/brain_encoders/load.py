from pathlib import Path
from typing import Literal, Union
from .dinov2_model import DINOV2_ROIS, DINOv2Encoder
from .fwrf_model import FWRF_ROIS, FWRFEncoder


def load_encoders(
	subject: int,
	roi: Union[str, list[str]],
	hemisphere: Literal["lh", "rh", "both"],
	berg_dir: Union[str, Path],
	device=None,
) -> dict:
	"""Every brain encoder with a model for the ROI(s), keyed by name ("fwrf", "dinov2").

	fwrf models cover both hemispheres, so `hemisphere` only applies to the DINOv2 encoder.
	"""
	rois = [roi] if isinstance(roi, str) else list(roi)
	encoders = {}
	if all(r in FWRF_ROIS for r in rois):
		encoders["fwrf"] = FWRFEncoder(subject, roi, berg_dir, device)
	else:
		print(f"Skipping fwrf encoder: no model for ROI(s) {[r for r in rois if r not in FWRF_ROIS]}")
	if all(r in DINOV2_ROIS for r in rois):
		encoders["dinov2"] = DINOv2Encoder(subject, roi, hemisphere, berg_dir, device)
	else:
		print(f"Skipping DINOv2 encoder: no model for ROI(s) {[r for r in rois if r not in DINOV2_ROIS]}")
	if not encoders:
		raise ValueError(f"No brain encoder has a model for ROI(s) {rois}")
	return encoders
