import torch
import numpy as np
from ...utils import resolve_device


class DepthEstimator:

	def __init__(self, device=None):
		self.device = resolve_device(device)
		torch.hub.list("intel-isl/MiDaS", trust_repo=True)
		self.zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, trust_repo=True)
		for module in self.zoe.modules():
			if hasattr(module, "drop_path1") and not hasattr(module, "drop_path"):
				module.drop_path = module.drop_path1
		self.zoe = self.zoe.to(self.device).eval()

	def compute(self, img):
		img_tensor = (
			torch.tensor(np.array(img).transpose(2, 0, 1))
			.unsqueeze(0)
			.float()
			.to(self.device)
			/ 255
		)
		with torch.no_grad():
			depth = self.zoe.infer(img_tensor).squeeze().detach().cpu().numpy()
			depth = depth[None,:,:]
		return depth
