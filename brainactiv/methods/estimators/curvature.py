import torch
import visualpriors
import numpy as np
from ...utils import resolve_device


class CurvatureEstimator:

	def __init__(self, device=None):
		self.device = resolve_device(device)

	def compute(self, img):
		img_tensor = torch.tensor(
			np.array(img.resize((256, 256))).transpose(2, 0, 1),
		).unsqueeze(0).float().to(self.device) / 255
		principal_curvature = (
			visualpriors.feature_readout(
				img_tensor * 2 - 1, "curvature", device=self.device,
			) / 2. + 0.5)[:,:2]
		principal_curvature = torch.nn.functional.interpolate(
			principal_curvature,
			size=img.size,
			mode="bicubic",
			align_corners=False,
		).squeeze(1).clamp(min=0, max=1)
		principal_curvature = principal_curvature.squeeze().permute(1,2,0).detach().cpu().numpy()
		gaussian_curvature = np.prod(principal_curvature, -1)[None,:,:]
		return gaussian_curvature
