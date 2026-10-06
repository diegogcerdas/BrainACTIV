import dataclasses
import types
import yaml  # type: ignore
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints


class BaseConfig:
	"""YAML (de)serialization shared by every config dataclass.

	Nested dataclass fields are built recursively from their YAML section and
	Path fields are converted from strings, so subclasses only declare fields.
	Unknown keys raise, which catches typos in config files early.
	"""

	@classmethod
	def from_yaml(cls, path: Union[str, Path]):
		with open(path) as fh:
			raw = yaml.safe_load(fh) or {}
		return cls.from_dict(raw)

	@classmethod
	def from_dict(cls, raw: dict):
		hints = get_type_hints(cls)
		names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
		unknown = set(raw) - names
		if unknown:
			raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
		kwargs = {k: _coerce(hints[k], v) for k, v in raw.items()}
		return cls(**kwargs)

	def to_dict(self) -> dict:
		return _plain(dataclasses.asdict(self))  # type: ignore[call-overload]

	def to_yaml(self) -> str:
		return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _coerce(hint, value):
	if value is None:
		return None
	# Optional[X] / X | None -> X
	if get_origin(hint) in (Union, types.UnionType):
		args = [a for a in get_args(hint) if a is not type(None)]
		if len(args) == 1:
			hint = args[0]
	if isinstance(hint, type) and issubclass(hint, BaseConfig):
		return hint.from_dict(value or {})
	if hint is Path:
		return Path(value).expanduser()
	return value


def _plain(obj):
	if isinstance(obj, dict):
		return {k: _plain(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [_plain(v) for v in obj]
	if isinstance(obj, Path):
		return str(obj)
	return obj


@dataclass
class PipelineConfig(BaseConfig):
	sd_model: str = "runwayml/stable-diffusion-v1-5"
	clip_model: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
	ip_adapter_checkpoint: Path = Path("checkpoints/ip-adapter_sd15.bin")
	projection_set_path: Path = Path("checkpoints/projection_set.npy")
	device: str = "auto"
	resolution: int = 512
	num_inference_steps: int = 50
	projection_temperature: float = 1e-2


@dataclass
class NSDConfig(BaseConfig):
	root: Path = Path("~/Documents/Datasets/NSD")
	tval_threshold: float = 5.0

	def __post_init__(self):
		self.root = Path(self.root).expanduser()


def _check_manipulation(subject: int, alpha: float, gamma: float, *hemispheres: str):
	"""Range checks shared by the image-manipulation experiments."""
	if subject not in range(1, 9):
		raise ValueError(f"subject must be in 1..8, got {subject}")
	if not 0.0 <= alpha <= 1.0:
		raise ValueError(f"alpha must be in [0, 1], got {alpha}")
	if not 0.0 < gamma <= 1.0:
		raise ValueError(f"gamma must be in (0, 1], got {gamma}")
	for hemisphere in hemispheres:
		if hemisphere not in ("lh", "rh", "both"):
			raise ValueError(f"hemisphere must be lh, rh or both, got {hemisphere!r}")


@dataclass
class ImageVariationConfig(BaseConfig):
	"""Experiment 1: manipulate images to maximize or minimize average ROI activation"""

	subject: int = 1
	roi: Union[str, list[str]] = "PPA"
	hemisphere: Literal["lh", "rh", "both"] = "both"
	do_maximization: bool = True
	use_other_subjects: bool = True
	alpha: float = 0.5
	gamma: float = 0.6
	do_projection: bool = True
	rng_seed: int = 0
	input_folder: Path = Path("examples")
	output_folder: Path = Path("outputs/exp1_image_variation")
	berg_dir: Path = Path("checkpoints/berg")
	nsd: NSDConfig = field(default_factory=NSDConfig)
	pipeline: PipelineConfig = field(default_factory=PipelineConfig)

	def __post_init__(self):
		_check_manipulation(self.subject, self.alpha, self.gamma, self.hemisphere)


@dataclass
class FeatureQuantificationConfig(BaseConfig):
	"""Experiment 2: quantify how image features change between original and manipulated images"""

	input_folder: Path = Path("examples")
	output_folder: Path = Path("outputs/exp1_image_variation")
	resolution: int = 64
	surface_normals_checkpoint: Path = Path("checkpoints/rgb2normal_consistency.pth")
	category_vectors_folder: Path = Path("checkpoints/category_vectors")
	device: str = "auto"


@dataclass
class ROIDifferencesConfig(BaseConfig):
	"""Experiment 3: manipulate images to accentuate one region over the other"""

	subject: int = 1
	roi1: Union[str, list[str]] = "PPA"
	hemisphere1: Literal["lh", "rh", "both"] = "both"
	roi2: Union[str, list[str]] = "OPA"
	hemisphere2: Literal["lh", "rh", "both"] = "both"
	do_maximization: bool = True
	use_other_subjects: bool = True
	alpha: float = 0.5
	gamma: float = 0.6
	do_projection: bool = True
	rng_seed: int = 0
	input_folder: Path = Path("examples")
	output_folder: Path = Path("outputs/exp3_roi_differences")
	berg_dir: Path = Path("checkpoints/berg")
	nsd: NSDConfig = field(default_factory=NSDConfig)
	pipeline: PipelineConfig = field(default_factory=PipelineConfig)

	def __post_init__(self):
		_check_manipulation(
			self.subject, self.alpha, self.gamma, self.hemisphere1, self.hemisphere2,
		)
