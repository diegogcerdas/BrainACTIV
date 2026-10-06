"""Compute the projection set: CLIP embeddings of natural images from LAION.

BrainACTIV projects modulation embeddings onto this set so that they stay close
to CLIP embeddings of real images. Writes <out>, (n_images, 1024) float16.

Images are streamed in random order from laion/relaion2B-en-research-safe, a
gated dataset: accept its terms on Hugging Face and run `hf auth login` first.
Dead links and images smaller than MIN_SIZE are skipped. Progress is saved every SAVE_EVERY images,
and re-running with the same arguments resumes from there.

Example:
python scripts/compute_projection_set.py --out checkpoints/projection_set.npy
"""

import argparse
import io
import json
import os
import sys
import traceback
import urllib.request
import datasets  # type: ignore
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm  # type: ignore
from brainactiv.methods import CLIP


DATASET = "laion/relaion2B-en-research-safe"
TIMEOUT = 2.5
MIN_SIZE = 64
SAVE_EVERY = 10_000


def fetch(url: str) -> Optional[Image.Image]:
	try:
		with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
			img = Image.open(io.BytesIO(r.read()))
			return img.convert("RGB") if min(img.size) >= MIN_SIZE else None
	except Exception:  # noqa: BLE001 -- dead links, timeouts and broken images are skipped
		return None


@torch.no_grad()
def embed(clip: CLIP, images: list[Image.Image]) -> np.ndarray:
	pixels = clip.processor(images=images, return_tensors="pt")["pixel_values"]
	return clip.clip(pixels.to(clip.device)).image_embeds.float().cpu().numpy()


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("--out", type=Path, default=Path("checkpoints/projection_set.npy"))
	p.add_argument("--n-images", type=int, default=400_000)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
	p.add_argument("--batch-size", type=int, default=64)
	p.add_argument("--workers", type=int, default=64, help="parallel image downloads")
	args = p.parse_args()
	if args.out.exists():
		raise SystemExit(f"{args.out} already exists; move it or pass another --out")

	# Partial results: the embeddings so far, plus how many dataset rows were used
	partial = args.out.with_name(args.out.stem + ".partial.npy")
	state_path = args.out.with_name(args.out.stem + ".partial.json")
	feats = np.zeros((args.n_images, 1024), dtype=np.float16)
	n_done, n_rows = 0, 0
	if partial.exists() and state_path.exists():
		state = json.loads(state_path.read_text())
		if state["seed"] != args.seed:
			raise SystemExit(f"{partial} used seed {state['seed']}; delete it or pass that seed")
		n_done, n_rows = state["n_done"], state["n_rows"]
		feats[:n_done] = np.load(partial)
		print(f">>> Resuming at {n_done} images ({n_rows} dataset rows used)")

	def save_partial():
		args.out.parent.mkdir(parents=True, exist_ok=True)
		np.save(partial, feats[:n_done])
		state_path.write_text(json.dumps({"seed": args.seed, "n_done": n_done, "n_rows": n_rows}))

	stream = datasets.load_dataset(DATASET, split="train", streaming=True)
	urls = (row["url"] for row in stream.shuffle(seed=args.seed).skip(n_rows))
	clip = CLIP(args.device)

	bar = tqdm(total=args.n_images, initial=n_done, desc="    embedding")
	with ThreadPoolExecutor(args.workers) as pool:

		def start_downloads():
			return [pool.submit(fetch, url) for url in islice(urls, 4 * args.batch_size)]

		pending = start_downloads()
		while pending and n_done < args.n_images:
			n_rows += len(pending)
			images = [f.result() for f in pending]
			images = [img for img in images if img is not None][:args.n_images - n_done]
			# Download the next chunk while this one is embedded
			pending = start_downloads()
			for start in range(0, len(images), args.batch_size):
				batch = embed(clip, images[start:start + args.batch_size])
				feats[n_done:n_done + len(batch)] = batch
				n_done += len(batch)
				bar.update(len(batch))
			if n_done // SAVE_EVERY > (n_done - len(images)) // SAVE_EVERY:
				save_partial()
	bar.close()

	np.save(args.out, feats[:n_done])
	partial.unlink(missing_ok=True)
	state_path.unlink(missing_ok=True)
	print(f">>> Done: {n_done} embeddings -> {args.out}")


if __name__ == "__main__":
	code = 0
	try:
		main()
	except SystemExit as e:
		if e.code not in (None, 0):
			print(e.code, file=sys.stderr)
			code = 1
	except BaseException:  # noqa: BLE001 -- includes Ctrl-C
		traceback.print_exc()
		code = 1
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(code)
