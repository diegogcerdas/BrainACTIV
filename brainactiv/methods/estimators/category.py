import numpy as np
from pathlib import Path


# Category vectors used in the paper (see scripts/compute_category_vectors.py)
CATEGORIES = [
	"faces",
	"hands",
	"feet",
	"people",
	"animals",
	"plants",
	"food",
	"furniture",
	"tools",
	"clothing",
	"electronics",
	"vehicles",
	"natural outdoors",
	"manmade outdoors",
	"manmade indoors",
	"text on an object",
]


class CategoryEstimator:

	def __init__(self, vectors_folder="checkpoints/category_vectors", categories=CATEGORIES):
		self.categories = list(categories)
		vectors = np.stack([np.load(Path(vectors_folder) / f"{c}.npy") for c in self.categories])
		self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

	def compute(self, clip_embeds):
		clip_embeds = np.asarray(clip_embeds, dtype=np.float64)
		clip_embeds = clip_embeds / np.linalg.norm(clip_embeds, axis=-1, keepdims=True)
		return clip_embeds @ self.vectors.T
