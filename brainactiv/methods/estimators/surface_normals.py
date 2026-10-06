import torch
import torch.nn as nn
import numpy as np
from ...utils import resolve_device


class SurfaceNormalEstimator:

	def __init__(self, model_path="checkpoints/rgb2normal_consistency.pth", device=None):
		self.device = resolve_device(device)
		self.normal_model = UNet()
		self.normal_model.load_state_dict(torch.load(model_path, map_location=self.device))
		self.normal_model = self.normal_model.to(self.device).eval()

	def compute(self, img):
		img_tensor = torch.tensor(
			np.array(img.resize((256, 256))).transpose(2, 0, 1),
			).unsqueeze(0).float().to(self.device) / 255
		with torch.no_grad():
			normal = self.normal_model(img_tensor)
			normal = torch.nn.functional.interpolate(
				normal,
				size=img.size,
				mode="bicubic",
				align_corners=False,
			).squeeze(1).clamp(min=0, max=1)
			normal = normal.squeeze().permute(1,2,0).detach().cpu().numpy()
			normal = np.moveaxis(normal, -1, 0)
		return normal

"""
The following is adapted from https://github.com/EPFL-VILAB/XTConsistency
Zamir, A. et al. Robust Learning Through Cross-Task Consistency. CVPR 2020.
"""

class UNet_up_block(nn.Module):
	def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
		super().__init__()
		self.up_sampling = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
		self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
		self.bn1 = nn.GroupNorm(8, output_channel)
		self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
		self.bn2 = nn.GroupNorm(8, output_channel)
		self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
		self.bn3 = nn.GroupNorm(8, output_channel)
		self.relu = torch.nn.ReLU()
		self.up_sample = up_sample

	def forward(self, prev_feature_map, x):
		if self.up_sample:
			x = self.up_sampling(x)
		x = torch.cat((x, prev_feature_map), dim=1)
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		x = self.relu(self.bn3(self.conv3(x)))
		return x

class UNet_down_block(nn.Module):
	def __init__(self, input_channel, output_channel, down_size=True):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
		self.bn1 = nn.GroupNorm(8, output_channel)
		self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
		self.bn2 = nn.GroupNorm(8, output_channel)
		self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
		self.bn3 = nn.GroupNorm(8, output_channel)
		self.max_pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()
		self.down_size = down_size

	def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		x = self.relu(self.bn3(self.conv3(x)))
		if self.down_size:
			x = self.max_pool(x)
		return x

class UNet(nn.Module):

	def __init__(self,  downsample=6, in_channels=3, out_channels=3):
		super().__init__()

		self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
		self.down1 = UNet_down_block(in_channels, 16, False)
		self.down_blocks = nn.ModuleList(
			[UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)],
		)

		bottleneck = 2**(4 + downsample)
		self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
		self.bn1 = nn.GroupNorm(8, bottleneck)
		self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
		self.bn2 = nn.GroupNorm(8, bottleneck)
		self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
		self.bn3 = nn.GroupNorm(8, bottleneck)

		self.up_blocks = nn.ModuleList(
			[UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)],
		)

		self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
		self.last_bn = nn.GroupNorm(8, 16)
		self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.down1(x)
		xvals = [x]
		for i in range(0, self.downsample):
			x = self.down_blocks[i](x)
			xvals.append(x)

		x = self.relu(self.bn1(self.mid_conv1(x)))
		x = self.relu(self.bn2(self.mid_conv2(x)))
		x = self.relu(self.bn3(self.mid_conv3(x)))

		for i in range(0, self.downsample)[::-1]:
			x = self.up_blocks[i](xvals[i], x)

		x = self.relu(self.last_bn(self.last_conv1(x)))
		x = self.relu(self.last_conv2(x))
		return x
