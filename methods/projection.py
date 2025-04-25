from sklearn.metrics import pairwise_distances
import numpy as np

def project_modulation_vector(features, modulation_vector, temp=1e-2):
    assert modulation_vector.ndim == 1
    cosines = 1 - pairwise_distances(features, modulation_vector.reshape(1,-1), metric='cosine').squeeze().astype(np.float128)
    exps = np.exp(cosines/temp)
    scores = exps / np.sum(exps)
    norms = np.linalg.norm(features, axis=1)
    directions = features / norms[:, None]
    modulation_vector = np.sum(scores*norms) * np.sum(scores[:,None]*directions, axis=0)
    modulation_vector = (modulation_vector / np.linalg.norm(modulation_vector)).astype(np.float32)
    return modulation_vector