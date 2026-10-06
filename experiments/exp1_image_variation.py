"""Experiment 1: manipulate images to maximize or minimize average ROI activation

Example:
python experiments/exp1_image_variation.py configs/exp1_image_variation.yaml
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from brainactiv.brain_encoders import load_encoders
from brainactiv.config import ImageVariationConfig
from brainactiv.pipeline import BrainACTIV, get_modulation_embedding
from brainactiv.datasets import NSD


def main(args):

	cfg = ImageVariationConfig.from_yaml(args.config)

	# Load BrainACTIV pipeline
	brainactiv = BrainACTIV.from_config(cfg.pipeline)

	# Load brain encoders (predict the subject's responses in the ROI)
	encoders = load_encoders(
		cfg.subject, cfg.roi, cfg.hemisphere, cfg.berg_dir, cfg.pipeline.device,
	)

	if not cfg.use_other_subjects:
		subjects = [cfg.subject]
	else:
		# Get modulation embed from rest of subjects
		subjects = [1,2,3,4,5,6,7,8]
		subjects.remove(cfg.subject)
	modulation_embed_list = []
	for s in subjects:
		try:
			dataset_train = NSD(
				root=str(cfg.nsd.root),
				subject=s,
				partition="train",
				hemisphere=cfg.hemisphere,
				roi=cfg.roi,
				tval_threshold=cfg.nsd.tval_threshold,
				return_trial_average=False,
			)
		except Exception as e:
			print(f"Failed to load dataset for subject {s}: {e}")
			continue

		# ==========================================================================================
		# If using your own NSD implementation,
		# you can replace the following with your own matrices:
		clip_features = dataset_train.get_clip_features()  # (num_images, clip_dim)
		activations = dataset_train.activations  # (num_images, num_reps, num_voxels)
		activations_trial_avg = np.nanmean(activations, axis=1)  # (num_images, num_voxels)
		activations_voxel_avg = activations_trial_avg.mean(axis=1)  # (num_images,)
		# ==========================================================================================

		mod_vec = get_modulation_embedding(
			clip_features,
			activations_voxel_avg,
		)
		modulation_embed_list.append(mod_vec)

	if not modulation_embed_list:
		raise RuntimeError(f"Could not load NSD data for any of subjects {subjects}")
	modulation_embed_stack = np.stack(modulation_embed_list, axis=0)
	modulation_embed = modulation_embed_stack.mean(axis=0)
	modulation_embed = modulation_embed / np.linalg.norm(modulation_embed)
	if not cfg.do_maximization:
		modulation_embed = -modulation_embed

	# Ensure output folder exists
	cfg.output_folder.mkdir(parents=True, exist_ok=True)

	# Get list of input images
	input_images = sorted(list(cfg.input_folder.glob("*.png")))
	acts_input = {name: [] for name in encoders}
	acts_output = {name: [] for name in encoders}

	# Produce list of seeds (one per image)
	num_images = len(input_images)
	seeds = [cfg.rng_seed + i for i in range(num_images)]

	# Manipulate each input image with BrainACTIV
	for i, image_path in enumerate(input_images):
		seed = seeds[i]
		image = Image.open(image_path)
		output_image = brainactiv.manipulate(
			image_ref=image,
			mod_embed=modulation_embed,
			alpha=cfg.alpha,
			gamma=cfg.gamma,
			seed=seed,
			do_projection=cfg.do_projection,
		)
		for name, encoder in encoders.items():
			acts_input[name].append(encoder.encode(image))
			acts_output[name].append(encoder.encode(output_image))
		output_path = cfg.output_folder / image_path.name
		output_image.save(output_path)

	# Save brain encoder activations for input and output images, per encoder
	for name in encoders:
		np.save(cfg.output_folder / f"acts_input_{name}.npy", np.stack(acts_input[name], axis=0))
		np.save(cfg.output_folder / f"acts_output_{name}.npy", np.stack(acts_output[name], axis=0))

	# Save configuration used for this experiment
	config_save_path = cfg.output_folder / "config.yaml"
	with open(config_save_path, "w") as f:
		f.write(cfg.to_yaml())

	# Save image order
	image_order_save_path = cfg.output_folder / "image_order.txt"
	with open(image_order_save_path, "w") as f:
		for image_path in input_images:
			f.write(f"{image_path.name}\n")

	# Summary: predicted ROI response averaged over voxels, per image
	print(f"\nPredicted {cfg.roi} response of subject {cfg.subject}, {num_images} images (z):")
	for name in encoders:
		before = np.stack(acts_input[name]).mean(axis=1)
		after = np.stack(acts_output[name]).mean(axis=1)
		change = after - before
		n_goal = int(((change > 0) if cfg.do_maximization else (change < 0)).sum())
		print(
			f"  {name:6s} input {before.mean():+.3f} -> output {after.mean():+.3f} | "
			f"change {change.mean():+.3f} ± {change.std():.3f} | "
			f"{'increased' if cfg.do_maximization else 'decreased'} in {n_goal}/{num_images} images"
		)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("config", type=Path)
	args = parser.parse_args()
	main(args)
