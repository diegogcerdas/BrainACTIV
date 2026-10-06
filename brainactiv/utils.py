import torch
from typing import Optional, Union


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
	"""Return `device` as a torch.device, picking cuda > mps > cpu for None or "auto"."""
	if device is None or device == "auto":
		if torch.cuda.is_available():
			device = "cuda:0"
		elif torch.backends.mps.is_available():
			device = "mps"
		else:
			device = "cpu"
	return torch.device(device)

def resize(measure, size):
	measure = torch.from_numpy(measure).float().unsqueeze(0)
	measure = torch.nn.functional.interpolate(
		measure,
		size=(size,size),
		mode="bilinear",
		align_corners=False,
	).squeeze(0)
	measure = measure.numpy()
	return measure
