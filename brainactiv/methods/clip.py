import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from ..utils import resolve_device


class CLIP(nn.Module):

	def __init__(self, device=None, clip_ckpt="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
		super().__init__()
		self.device = resolve_device(device)
		self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_ckpt).to(self.device)
		self.processor = CLIPProcessor.from_pretrained(clip_ckpt)

	@torch.no_grad()
	def forward(self, img):
		input = self.processor(
			images=img,
			return_tensors="pt",
			padding=True,
		)["pixel_values"].to(self.device)
		output = self.clip(input).image_embeds[0]
		return output
