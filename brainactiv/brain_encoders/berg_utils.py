import urllib.request
import numpy as np
from pathlib import Path
from typing import Union
from PIL import Image


BERG_BASE = "https://brain-encoding-response-generator.s3.amazonaws.com"


def download_berg_file(berg_dir: Union[str, Path], key: str) -> Path:
	"""Path of `key` inside berg_dir (BERG's S3 layout), downloading it on first use.

	BERG itself expects berg_dir to be populated already (e.g. with `aws s3 sync`), so the
	encoders fetch only the files their model needs.
	"""
	path = Path(berg_dir) / key
	if not path.exists():
		path.parent.mkdir(parents=True, exist_ok=True)
		print(f"Downloading BERG file {path.name}")
		tmp = path.with_name(path.name + ".part")
		urllib.request.urlretrieve(f"{BERG_BASE}/{key}", tmp)
		tmp.rename(path)
	return path


def to_stimulus(image: Image.Image) -> np.ndarray:
	"""(1, 3, H, W) uint8 array, the input format of BERG's image models."""
	array = np.array(image.convert("RGB"))
	if array.shape[0] != array.shape[1]:
		raise ValueError("BERG encoding models expect square images")
	return array.transpose(2, 0, 1)[None]
