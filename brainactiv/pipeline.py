import torch
import numpy as np
from typing import Optional
from PIL import Image
from pathlib import Path
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.linear_model import RidgeCV  # type: ignore
from scipy.stats import zscore  # type: ignore
from ip_adapter import IPAdapter  # type: ignore
from .methods.clip import CLIP
from .methods.slerp import slerp
from .config import PipelineConfig
from .utils import resolve_device


class BrainACTIV:

	def __init__(
		self,
		projection_set_path: Optional[Path] = Path("checkpoints/projection_set.npy"),
		ip_adapter_checkpoint: Path = Path("checkpoints/ip-adapter_sd15.bin"),
		device: Optional[str] = None,
		sd_model: str = "runwayml/stable-diffusion-v1-5",
		clip_model: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
		resolution: int = 512,
		num_inference_steps: int = 50,
		projection_temperature: float = 1e-2,
	):

		self.device = resolve_device(device)
		self.num_inference_steps = num_inference_steps
		self.projection_temperature = projection_temperature

		# Initialize Stable Diffusion, IP-Adapter, and CLIP model
		self.resolution = (resolution, resolution)
		self.diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
			sd_model,
			torch_dtype=torch.float16,
		).to(self.device)
		self.diffusion_pipeline.scheduler = DDIMScheduler.from_config(
			self.diffusion_pipeline.scheduler.config,
		)
		self.diffusion_pipeline.safety_checker = None
		self.ip_model = IPAdapter(
			self.diffusion_pipeline,
			clip_model,
			str(ip_adapter_checkpoint),
			self.device,
		)
		self.clip_extractor = CLIP(self.device, clip_model)

		# Load set for projection to CLIP image space (None if embeddings are already projected)
		self.projection_set = None if projection_set_path is None else np.load(projection_set_path)

	@classmethod
	def from_config(cls, cfg: PipelineConfig) -> "BrainACTIV":
		return cls(
			projection_set_path=cfg.projection_set_path,
			ip_adapter_checkpoint=cfg.ip_adapter_checkpoint,
			device=cfg.device,
			sd_model=cfg.sd_model,
			clip_model=cfg.clip_model,
			resolution=cfg.resolution,
			num_inference_steps=cfg.num_inference_steps,
			projection_temperature=cfg.projection_temperature,
		)

	def project_modulation_embedding(
		self,
		mod_embed: np.ndarray,
	) -> np.ndarray:
		if self.projection_set is None:
			raise ValueError("Projection needs a projection set; pass projection_set_path")
		return project_embedding(mod_embed, self.projection_set, self.projection_temperature)

	def manipulate(
		self,
		image_ref: Image.Image,
		mod_embed: np.ndarray,
		alpha: float,
		gamma: float,
		seed: int,
		do_projection: bool,
	) -> Image.Image:

		if do_projection:
			mod_embed = self.project_modulation_embedding(mod_embed)

		image_ref = image_ref.convert("RGB").resize(self.resolution)
		image_ref_clip = self.clip_extractor(image_ref).detach().cpu().numpy()

		endpoint = mod_embed * np.linalg.norm(image_ref_clip)
		embeds = torch.from_numpy(
			slerp(image_ref_clip, endpoint, 1, t0=alpha, t1=alpha),
		).unsqueeze(1).float().to(self.device)[0]

		with torch.no_grad():
			image_new = self.ip_model.generate(
				clip_image_embeds=embeds,
				image=image_ref,
				strength=gamma,
				num_samples=1,
				num_inference_steps=self.num_inference_steps,
				seed=seed,
			)[0]

		return image_new


def project_embedding(
	mod_embed: np.ndarray,
	projection_set: np.ndarray,
	temperature: float = 1e-2,
) -> np.ndarray:
	assert mod_embed.ndim == 1
	cosines = 1 - pairwise_distances(
		projection_set,
		mod_embed.reshape(1, -1),
		metric="cosine",
	).squeeze().astype(np.float32)
	exps = np.exp(cosines / temperature)
	scores = exps / np.sum(exps)
	norms = np.linalg.norm(projection_set, axis=1)
	directions = projection_set / norms[:, None]
	mod_embed = np.sum(scores * norms) * np.sum(scores[:, None] * directions, axis=0)
	return (mod_embed / np.linalg.norm(mod_embed)).astype(np.float32)


def get_modulation_embedding(
	clip_features: np.ndarray,
	activations: np.ndarray,
	alphas: np.ndarray = np.logspace(-1, 5, 13),
) -> np.ndarray:
	X_train = clip_features / (np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8)
	Y_train = zscore(activations, axis=0)
	modulation_vector = RidgeCV(alphas=alphas).fit(X_train, Y_train).coef_
	modulation_vector = modulation_vector / np.linalg.norm(modulation_vector)
	return modulation_vector


def get_accentuation_embedding(
	clip_features: np.ndarray,
	activations1: np.ndarray,
	activations2: np.ndarray,
	alphas: np.ndarray = np.logspace(-1, 5, 13),
) -> np.ndarray:
	X_train = clip_features / (np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8)
	Y_train = zscore(activations1, axis=0) - zscore(activations2, axis=0)
	Y_train = zscore(Y_train, axis=0)
	modulation_vector = RidgeCV(alphas=alphas).fit(X_train, Y_train).coef_
	modulation_vector = modulation_vector / np.linalg.norm(modulation_vector)
	return modulation_vector
