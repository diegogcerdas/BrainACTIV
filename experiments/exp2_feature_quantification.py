"""Experiment 2: quantify how image features change between original and manipulated images

Features are computed for every image in output_folder (e.g. exp1 results) and for the image
with the same name in input_folder (the originals). The differences (output - input) are saved
to output_folder as feature_diff_*.npy.

Example:
python experiments/exp2_feature_quantification.py configs/exp2_feature_quantification.yaml
"""

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm  # type: ignore
from pathlib import Path

from brainactiv.config import FeatureQuantificationConfig
from brainactiv.utils import resize
from brainactiv.methods import CLIP
from brainactiv.methods.estimators import (
	CategoryEstimator,
	DepthEstimator,
	SurfaceNormalEstimator,
	CurvatureEstimator,
	compute_brightness,
	compute_entropy,
	compute_saturation,
	compute_warmth,
)

SPATIAL_FEATURES = ["depths", "curvatures", "warmths", "saturations", "brightnesses", "entropies"]


def main(args):

	cfg = FeatureQuantificationConfig.from_yaml(args.config)
	res = cfg.resolution

	clip = CLIP(cfg.device)
	depth_estimator = DepthEstimator(cfg.device)
	normal_estimator = SurfaceNormalEstimator(cfg.surface_normals_checkpoint, cfg.device)
	curvature_estimator = CurvatureEstimator(cfg.device)
	category_estimator = CategoryEstimator(cfg.category_vectors_folder)

	def compute_features(image: Image.Image) -> dict:
		clip_features = clip(image).cpu().numpy()
		return {
			"clip_features": clip_features,
			"categories": category_estimator.compute(clip_features),
			"depths": resize(depth_estimator.compute(image), res)[0],
			"normals": resize(normal_estimator.compute(image), res),  # x, y, z components
			"curvatures": resize(curvature_estimator.compute(image), res)[0],
			"warmths": resize(compute_warmth(image), res)[0],
			"saturations": resize(compute_saturation(image), res)[0],
			"brightnesses": resize(compute_brightness(image), res)[0],
			"entropies": resize(compute_entropy(image), res)[0],
		}

	# Match each manipulated image to the original with the same name
	output_images = sorted(cfg.output_folder.glob("*.png"))
	missing = [p.name for p in output_images if not (cfg.input_folder / p.name).exists()]
	if missing:
		raise FileNotFoundError(f"No original image in {cfg.input_folder} for: {missing}")

	# Compute features for both images and take the difference (output - input)
	features_input, features_output = [], []
	for image_path in tqdm(output_images):
		image_input = Image.open(cfg.input_folder / image_path.name).convert("RGB")
		image_output = Image.open(image_path).convert("RGB")
		features_input.append(compute_features(image_input))
		features_output.append(compute_features(image_output))
	diffs = {
		name: np.stack([o[name] - i[name] for i, o in zip(features_input, features_output)])
		for name in features_input[0]
	}

	# Save the feature differences next to the manipulated images
	for name, diff in diffs.items():
		np.save(cfg.output_folder / f"feature_diff_{name}.npy", diff)

	# Save configuration used for this experiment
	with open(cfg.output_folder / "feature_diff_config.yaml", "w") as f:
		f.write(cfg.to_yaml())

	# Save image order (row order of the feature arrays)
	with open(cfg.output_folder / "feature_diff_image_order.txt", "w") as f:
		for image_path in output_images:
			f.write(f"{image_path.name}\n")

	# Save category order (column order of feature_diff_categories.npy)
	with open(cfg.output_folder / "feature_diff_category_names.txt", "w") as f:
		for category in category_estimator.categories:
			f.write(f"{category}\n")

	# Summary: mean change across images
	clip_input = np.stack([f["clip_features"] for f in features_input])
	clip_output = np.stack([f["clip_features"] for f in features_output])
	clip_cos = (clip_input * clip_output).sum(1) / (
		np.linalg.norm(clip_input, axis=1) * np.linalg.norm(clip_output, axis=1)
	)
	print(f"\nMean change across {len(output_images)} images (output - input):")
	print(f"  CLIP cosine similarity (input vs output): {clip_cos.mean():.3f}")
	print("  Categories (cosine similarity to category vector):")
	category_diffs = diffs["categories"].mean(0)
	for k in np.argsort(category_diffs)[::-1]:
		print(f"    {category_estimator.categories[k]:18s} {category_diffs[k]:+.4f}")
	print("  Spatial features (mean over pixels):")
	for name in SPATIAL_FEATURES:
		print(f"    {name:18s} {diffs[name].mean():+.4f}")
	for axis, value in zip("xyz", diffs["normals"].mean(axis=(0, 2, 3))):
		print(f"    {'normals ' + axis:18s} {value:+.4f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("config", type=Path)
	args = parser.parse_args()
	main(args)
