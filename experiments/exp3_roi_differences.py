"""Experiment 3: manipulate images to accentuate one region over the other

Example:
python experiments/exp3_roi_differences.py configs/exp3_roi_differences.yaml
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from brainactiv.brain_encoders import load_encoders
from brainactiv.config import ROIDifferencesConfig
from brainactiv.pipeline import BrainACTIV, get_accentuation_embedding
from brainactiv.datasets import NSD


def main(args):

	cfg = ROIDifferencesConfig.from_yaml(args.config)

	# Load BrainACTIV pipeline
	brainactiv = BrainACTIV.from_config(cfg.pipeline)

	# Load brain encoders (predict the subject's responses in each ROI)
	encoders = {
		"roi1": load_encoders(
			cfg.subject, cfg.roi1, cfg.hemisphere1, cfg.berg_dir, cfg.pipeline.device,
		),
		"roi2": load_encoders(
			cfg.subject, cfg.roi2, cfg.hemisphere2, cfg.berg_dir, cfg.pipeline.device,
		),
	}

	if not cfg.use_other_subjects:
		subjects = [cfg.subject]
	else:
		# Get modulation embed from rest of subjects
		subjects = [1,2,3,4,5,6,7,8]
		subjects.remove(cfg.subject)
	modulation_embed_list = []
	for s in subjects:

		try:
			dataset_train_1 = NSD(
				root=str(cfg.nsd.root),
				subject=s,
				partition="train",
				hemisphere=cfg.hemisphere1,
				roi=cfg.roi1,
				tval_threshold=cfg.nsd.tval_threshold,
				return_trial_average=False,
			)
			dataset_train_2 = NSD(
				root=str(cfg.nsd.root),
				subject=s,
				partition="train",
				hemisphere=cfg.hemisphere2,
				roi=cfg.roi2,
				tval_threshold=cfg.nsd.tval_threshold,
				return_trial_average=False,
			)
		except Exception as e:
			print(f"Failed to load dataset for subject {s}: {e}")
			continue

		# ==========================================================================================
		# If using your own NSD implementation,
		# you can replace the following with your own matrices:
		clip_features = dataset_train_1.get_clip_features()  # (num_images, clip_dim)
		activations = dataset_train_1.activations  # (num_images, num_reps, num_voxels)
		activations_trial_avg = np.nanmean(activations, axis=1)  # (num_images, num_voxels)
		activations_voxel_avg1 = activations_trial_avg.mean(axis=1)  # (num_images,)
		activations = dataset_train_2.activations  # (num_images, num_reps, num_voxels)
		activations_trial_avg = np.nanmean(activations, axis=1)  # (num_images, num_voxels)
		activations_voxel_avg2 = activations_trial_avg.mean(axis=1)  # (num_images,)
		# ==========================================================================================

		mod_vec = get_accentuation_embedding(
			clip_features,
			activations_voxel_avg1,
			activations_voxel_avg2,
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
	acts_input = {(r, name): [] for r in encoders for name in encoders[r]}
	acts_output = {(r, name): [] for r in encoders for name in encoders[r]}

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
		for r in encoders:
			for name, encoder in encoders[r].items():
				acts_input[r, name].append(encoder.encode(image))
				acts_output[r, name].append(encoder.encode(output_image))
		output_path = cfg.output_folder / image_path.name
		output_image.save(output_path)

	# Save brain encoder activations for input and output images, per ROI and encoder
	for r, name in acts_input:
		np.save(cfg.output_folder / f"acts_input_{r}_{name}.npy", np.stack(acts_input[r, name]))
		np.save(cfg.output_folder / f"acts_output_{r}_{name}.npy", np.stack(acts_output[r, name]))

	# Save configuration used for this experiment
	config_save_path = cfg.output_folder / "config.yaml"
	with open(config_save_path, "w") as f:
		f.write(cfg.to_yaml())

	# Save image order
	image_order_save_path = cfg.output_folder / "image_order.txt"
	with open(image_order_save_path, "w") as f:
		for image_path in input_images:
			f.write(f"{image_path.name}\n")

	# Summary: change in each ROI's predicted response (averaged over voxels) and in roi1 - roi2
	print(
		f"\nPredicted response change of subject {cfg.subject}, {num_images} images (z), "
		f"roi1 = {cfg.roi1}, roi2 = {cfg.roi2}:"
	)
	names = list(dict.fromkeys([*encoders["roi1"], *encoders["roi2"]]))
	for name in names:
		change = {
			r: np.stack(acts_output[r, name]).mean(axis=1) - np.stack(acts_input[r, name]).mean(axis=1)
			for r in encoders if name in encoders[r]
		}
		line = " | ".join(f"{r} {c.mean():+.3f} ± {c.std():.3f}" for r, c in change.items())
		if len(change) == 2:
			diff = change["roi1"] - change["roi2"]
			n_goal = int(((diff > 0) if cfg.do_maximization else (diff < 0)).sum())
			line += (
				f" | roi1 - roi2 {diff.mean():+.3f} ± {diff.std():.3f}, "
				f"{'increased' if cfg.do_maximization else 'decreased'} in {n_goal}/{num_images} images"
			)
		print(f"  {name:6s} {line}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("config", type=Path)
	args = parser.parse_args()
	main(args)
