"""Compute the CLIP category vectors used to quantify object/scene categories in images.

As detailed in Appendix A.1 of the paper:
  1. Nouns: every hyponym of SYNSETS in WordNet 3.0 whose name only contains
     dictionary words (hunspell en_US 2018.04.16), 17,086 rows.
  2. Scores: zero-shot classification of each noun's description
     ("name: definition") into the CATEGORIES labels with facebook/bart-large-mnli.
  3. Embeddings: CLIP text embedding of each description, averaged over the 18
     TEMPLATES and normalized.
  4. Vectors: ridge regression (lambda=100) from the embeddings to the scores,
     weighted by each category's salience. Text is instead represented by the
     templated embedding of TEXT_PHRASE.

Writes <out>/{category}.npy (1024,) unit vectors. Intermediate results are kept
in <cache> (wordnet.csv, wordnet_scores.csv, wordnet_vecs.npy) and reused if present.

Example:
python scripts/compute_category_vectors.py \
	--out checkpoints/category_vectors \
	--cache checkpoints/wordnet
"""

import argparse
import io
import tarfile
import urllib.request
import nltk  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import torch
from pathlib import Path
from spylls.hunspell import Dictionary  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, pipeline
from brainactiv.utils import resolve_device


CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLASSIFIER = "facebook/bart-large-mnli"

DICTIONARY_URL = (
	"http://archive.ubuntu.com/ubuntu/pool/main/s/scowl/hunspell-en-us_2018.04.16-1_all.deb"
)

SYNSETS = [
	"placental.n.01", "amphibian.n.03", "bird.n.01", "fish.n.01", "reptile.n.01", "person.n.01",
	"external_body_part.n.01", "food.n.02", "plant.n.02", "body_of_water.n.01",
	"geological_formation.n.01", "land.n.04", "building.n.01", "room.n.01", "way.n.06",
	"facility.n.01", "vehicle.n.01", "commodity.n.01", "instrumentality.n.03", "plaything.n.01",
	"article.n.02", "publication.n.01", "sign.n.02", "correspondence.n.01", "written_record.n.01",
]

# Category name -> zero-shot label
CATEGORIES = {
	"faces": "related to faces, eyes, nose, mouth",
	"hands": "related to hands, arms, fingers",
	"feet": "related to feet, legs, toes",
	"people": "related to people, humans, persons",
	"animals": "related to animals, creatures, fauna",
	"plants": "related to plants, greenery, flora",
	"food": "related to food, meals, eating",
	"furniture": "related to furniture, household items",
	"tools": "related to tools, equipment, instruments",
	"clothing": "related to clothing, textiles, garments",
	"electronics": "related to electronics, gadgets, devices",
	"vehicles": "related to vehicles, transportation, travel",
	"text": "related to written text, signs",
	"natural outdoors": "related to natural areas, landscapes, outdoors",
	"manmade outdoors": "related to urban areas, buildings, structures",
	"manmade indoors": "related to indoors, rooms, interiors",
}
TEXT_PHRASE = "text on an object"
RIDGE_LAMBDA = 100

# CLIP's ImageNet prompt templates (Radford et al., 2021)
TEMPLATES = [
	"a photo of a {}.",
	"a blurry photo of a {}.",
	"a black and white photo of a {}.",
	"a low contrast photo of a {}.",
	"a high contrast photo of a {}.",
	"a bad photo of a {}.",
	"a good photo of a {}.",
	"a photo of a small {}.",
	"a photo of a big {}.",
	"a photo of the {}.",
	"a blurry photo of the {}.",
	"a black and white photo of the {}.",
	"a low contrast photo of the {}.",
	"a high contrast photo of the {}.",
	"a bad photo of the {}.",
	"a good photo of the {}.",
	"a photo of the small {}.",
	"a photo of the big {}.",
]


def load_dictionary(cache: Path) -> Dictionary:
	"""hunspell en_US, extracted from the Ubuntu package on first use."""
	base = cache / "hunspell" / "en_US"
	if not base.with_suffix(".dic").exists():
		base.parent.mkdir(parents=True, exist_ok=True)
		with urllib.request.urlopen(DICTIONARY_URL, timeout=120) as r:
			deb = r.read()
		pos = 8
		while pos < len(deb):
			name = deb[pos:pos + 16].decode().strip().rstrip("/")
			size = int(deb[pos + 48:pos + 58])
			if name.startswith("data.tar"):
				data = deb[pos + 60:pos + 60 + size]
				break
			pos += 60 + size + size % 2
		with tarfile.open(fileobj=io.BytesIO(data)) as tar:
			for ext in (".aff", ".dic"):
				member = tar.extractfile(f"./usr/share/hunspell/en_US{ext}")
				base.with_suffix(ext).write_bytes(member.read())
	return Dictionary.from_files(str(base))


def build_nouns(cache: Path) -> pd.DataFrame:
	"""One row per (hypernym, hyponym); synsets under two hypernyms appear twice."""
	nltk.download("wordnet", quiet=True)
	from nltk.corpus import wordnet as wn  # type: ignore

	dictionary = load_dictionary(cache)
	rows = []
	for hypernym in SYNSETS:
		root = wn.synset(hypernym)
		for synset in set(root.closure(lambda s: s.hyponyms())) | {root}:
			name = synset.name().split(".")[0].replace("_", " ")
			if all(dictionary.lookup(word) for word in name.split()):
				rows.append((hypernym, synset.name(), name, f"{name}: {synset.definition()}"))
	nouns = pd.DataFrame(rows, columns=["hypernym", "identifier", "name", "description"])
	return nouns.sort_values(["name", "identifier", "hypernym"]).reset_index(drop=True)


def classify(descriptions: list[str], device: torch.device) -> np.ndarray:
	"""(n_descriptions, n_categories) zero-shot probabilities; each row sums to 1."""
	classifier = pipeline("zero-shot-classification", model=CLASSIFIER, device=device)
	labels = list(CATEGORIES.values())
	scores = np.empty((len(descriptions), len(labels)), dtype=np.float32)
	for i, description in enumerate(tqdm(descriptions, desc="    classifying")):
		result = classifier(description, labels, multi_label=False)
		by_label = dict(zip(result["labels"], result["scores"]))
		scores[i] = [by_label[label] for label in labels]
	return scores


@torch.no_grad()
def embed(texts: list[str], device: torch.device, batch_size: int = 16) -> np.ndarray:
	"""(n_texts, 1024) CLIP text embeddings averaged over TEMPLATES, normalized."""
	tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
	model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
	out = np.empty((len(texts), 1024), dtype=np.float32)
	for start in tqdm(range(0, len(texts), batch_size), desc="    embedding"):
		batch = texts[start:start + batch_size]
		prompts = [template.format(text) for text in batch for template in TEMPLATES]
		tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
		feats = model(**tokens.to(device)).text_embeds.float().cpu().numpy()
		feats = feats.reshape(len(batch), len(TEMPLATES), -1).mean(axis=1)
		out[start:start + len(batch)] = feats / np.linalg.norm(feats, axis=1, keepdims=True)
	return out


def fit_category_vectors(vecs: np.ndarray, scores: np.ndarray) -> np.ndarray:
	"""(1024, n_categories) unit vectors from salience-weighted ridge regression."""
	salience = scores / scores.sum(axis=1, keepdims=True)
	targets = scores * salience
	X = vecs.astype(np.float64)
	W = np.linalg.solve(X.T @ X + RIDGE_LAMBDA * np.eye(X.shape[1]), X.T @ targets)
	return W / np.linalg.norm(W, axis=0, keepdims=True)


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("--out", type=Path, default=Path("checkpoints/category_vectors"))
	p.add_argument("--cache", type=Path, default=Path("checkpoints/wordnet"))
	p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
	args = p.parse_args()
	device = resolve_device(args.device)
	args.cache.mkdir(parents=True, exist_ok=True)
	nouns_path = args.cache / "wordnet.csv"
	scores_path = args.cache / "wordnet_scores.csv"
	vecs_path = args.cache / "wordnet_vecs.npy"

	if nouns_path.exists():
		nouns = pd.read_csv(nouns_path)
	else:
		print(">>> Building noun list from WordNet")
		nouns = build_nouns(args.cache)
		nouns.to_csv(nouns_path, index=False)
	print(f">>> {len(nouns)} nouns")

	# Classification and embedding only depend on the synset, so run them once per
	# identifier and store the results in the same row order
	ids = list(dict.fromkeys(nouns.identifier))
	descriptions = nouns.drop_duplicates("identifier").set_index("identifier").description
	if scores_path.exists():
		scores = pd.read_csv(scores_path).drop_duplicates("identifier").set_index("identifier")
	else:
		print(">>> Classifying descriptions")
		probs = classify([descriptions[i] for i in ids], device)
		scores = pd.DataFrame(probs, index=pd.Index(ids, name="identifier"))
		scores.columns = list(CATEGORIES.values())
		scores.loc[nouns.identifier].reset_index().to_csv(scores_path, index=False)
	if vecs_path.exists():
		vecs = np.load(vecs_path)
		if len(vecs) != len(nouns):
			raise SystemExit(f"{vecs_path} has {len(vecs)} rows but {nouns_path} has {len(nouns)}")
	else:
		print(">>> Embedding descriptions")
		unique_vecs = dict(zip(ids, embed([descriptions[i] for i in ids], device)))
		vecs = np.stack([unique_vecs[i] for i in nouns.identifier])
		np.save(vecs_path, vecs)

	# wordnet_vecs.npy rows follow wordnet_scores.csv, so align by identifier
	order = pd.read_csv(scores_path).identifier
	probs = scores.loc[order, list(CATEGORIES.values())].values
	vectors = fit_category_vectors(vecs, probs)

	args.out.mkdir(parents=True, exist_ok=True)
	for i, category in enumerate(CATEGORIES):
		if category != "text":
			np.save(args.out / f"{category}.npy", vectors[:, i])
	np.save(args.out / f"{TEXT_PHRASE}.npy", embed([TEXT_PHRASE], device)[0])
	print(f">>> Done: {len(CATEGORIES)} category vectors -> {args.out}")


if __name__ == "__main__":
	main()
