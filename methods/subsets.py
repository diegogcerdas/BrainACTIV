import json
import os

import numpy as np

supercategories = {
    'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    'outdoor': ['traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench'],
    'animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'wild_animal': ['horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'accessory': ['hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase'],
    'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    'kitchen': ['bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
    'food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
    'furniture': ['chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door'],
    'electronic': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
    'appliance': ['microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender'],
    'indoor': ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'],
}

subsets = {
    'wild_animals': {
        'all_positives': [['wild_animal']],
        'negatives': ['person', 'vehicle', 'food']
    },
    'vehicles': {
        'all_positives': [['vehicle']],
        'negatives': ['person', 'animal', 'food']
    },
    'sports': {
        'all_positives': [['person', 'sports']],
        'negatives': ['animal', 'vehicle', 'food']
    },
    'food': {
        'all_positives': [['food']],
        'negatives': ['person', 'animal', 'vehicle']
    },
    'birds': {
        'all_positives': [['bird']],
        'negatives': ['person', 'food', 'vehicle', 'wild_animal', 'cat', 'dog']
    },
    'furniture': {
        'all_positives': [['furniture']],
        'negatives': ['person', 'food', 'vehicle', 'animal']
    },
}

SUBSETS = list(subsets.keys())

def get_subset_indices(nsd, subset):
    assert subset in SUBSETS
    all_positives = subsets[subset]['all_positives']
    negatives = subsets[subset]['negatives']
    all_indices = set()
    # Load category search dictionaries
    f = os.path.join(nsd.root, 'category2coco_ids.json')
    category2coco_ids = json.load(open(f))
    f = os.path.join(nsd.root, 'coco_id2categories.json')
    coco_id2categories = json.load(open(f))
    # Get indices for each supercategory
    for cat, elements in supercategories.items():
        idxs = [category2coco_ids.get(el, []) for el in elements]
        idxs = np.unique(np.concatenate(idxs)).astype(int).tolist()
        category2coco_ids[cat] = idxs
        for idx in idxs:
            coco_id2categories[str(idx)].append(cat)
    # For each list of categories in all_positives...
    for positives in all_positives:
        # Get indices shared by all categories in the list
        shared = set(category2coco_ids[positives[0]])
        for positive in positives:
            shared = set.intersection(shared, set(category2coco_ids[positive])) 
        # Only keep relevant indices for subject
        shared = shared.intersection(set(nsd.coco_ids))
        # For each index in the shared set...
        for idx in shared:
            categories = coco_id2categories[str(idx)]
            # If none of the negatives are in the corresponding image, add the index
            if not any([c in negatives for c in categories]): all_indices.add(idx)
    all_indices = np.array(list(all_indices))
    return all_indices