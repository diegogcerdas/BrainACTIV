import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch.utils import data
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from dataset.nsd import NaturalScenesDataset


class NSDCLIPFeaturesDataset(data.Dataset):
    def __init__(
        self,
        nsd: NaturalScenesDataset,
    ):
        super().__init__()
        self.nsd = nsd
        self.features = self.load_features()
        self.features = self.features[self.nsd.coco_id_mask]

    def __len__(self):
        return len(self.nsd)

    def __getitem__(self, idx):
        features = self.features[idx]
        if self.nsd.return_activations:
            img, activation, _ = self.nsd[idx]
            return img, features, activation
        img, _ = self.nsd[idx]
        return img, features

    def load_features(self):
        f = os.path.join(self.nsd.subj_dir, "clip_features.npy")
        if not os.path.exists(f):
            print("Computing features...")
            clip_extractor = CLIPExtractor()
            features = clip_extractor.extract_for_dataset(self.nsd)
            np.save(f, features)
            print("Done.")
        else:
            features = np.load(f).astype(np.float32)
        return features
    
    def get_modulation_vector(self, lmbda=10):
        assert self.nsd.return_activations
        Y_train = self.nsd.activations
        X_train = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        W = X_train.T @ X_train + lmbda * np.eye(X_train.shape[1])
        W = (np.linalg.inv(W) @ X_train.T @ Y_train).T
        modulation_vector = W / np.linalg.norm(W, axis=(1 if len(W.shape) > 1 else 0), keepdims=True)
        return modulation_vector
    
class CLIPExtractor(nn.Module):
    def __init__(self, device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")):
        super().__init__()
        self.device = device
        self.clip = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').to(device)
        self.processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    def extract_for_dataset(self, dataset: NaturalScenesDataset):
        assert dataset.partition == "all"
        assert dataset.transform is None
        features = []
        for i in tqdm(range(len(dataset))):
            img = dataset[i][0]
            x = self(img).detach().cpu().numpy()
            features.append(x)
        features = np.stack(features, axis=0).astype(np.float32)
        return features

    @torch.no_grad()
    def forward(self, img: Image.Image):
        input = self.processor(images=img, return_tensors="pt", padding=True)['pixel_values'].to(self.device)
        output = self.clip(input).image_embeds[0]
        return output
