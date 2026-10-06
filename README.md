<h1 align="center">
  <b>BrainACTIV: Identifying visuo-semantic properties driving cortical selectivity using diffusion-based image manipulation</b><br>
</h1>

**Brain Activation Control Through Image Variation (BrainACTIV)** is a method for manipulating a reference image to **enhance or decrease activity in a target cortical region** using pretrained diffusion models. The manipulation of a reference image allows for fine-grained and reliable offline identification of **optimal visuo-semantic properties**, as well as producing **controlled stimuli for novel neuroimaging studies**.

<div align='center'>

[**BrainACTIV: Identifying visuo-semantic properties driving cortical selectivity using diffusion-based image manipulation**](https://www.biorxiv.org/content/10.1101/2024.10.29.620889)<br>
Diego García Cerdas, Christina Sartzetaki, Magnus Petersen, Gemma Roig, Pascal Mettes and Iris Groen<br>
<strong> International Conference on Learning Representations (ICLR) 2025</strong>

<p align="center">
<a href="https://diegogcerdas.github.io/BrainACTIV/"><img src="https://img.shields.io/badge/Project%20Page-lightgray?style=for-the-badge"></a>
<a href="https://openreview.net/forum?id=CGON8Btleu"><img src="https://img.shields.io/badge/ICLR Paper-darkred?style=for-the-badge"></a>
  <a href="https://www.biorxiv.org/content/10.1101/2024.10.29.620889"><img src="https://img.shields.io/badge/bioRxiv Preprint-red?style=for-the-badge&labelColor=%23CC0000&color=%23000000"></a>

</p>

<div align='left'>

## Try Out Image Manipulation

<p align="left">
  <a href="https://colab.research.google.com/drive/1fL9q6PaOZ1S2gvXab2dzeKCBL88vWER_?usp=sharing"><img src="https://img.shields.io/badge/Google_Colab-yellow?style=for-the-badge&logo=googlecolab&labelColor=000000"></a>
</p>

You can try BrainACTIV on your own images through the Google Colab notebook linked above (also available as [`BrainACTIV_Demo.ipynb`](BrainACTIV_Demo.ipynb)). Besides the manipulated image, it shows the predicted activation of the target region from two brain encoders, and how the presence of 16 object and scene categories changes.

**Note:** Due to storage constraints, the notebook only provides Subject 1's modulation embeddings (computed from the rest of subjects, and already projected to CLIP image space).

## Manipulate images from the Natural Scenes Dataset (NSD)

### Installation
```bash
conda create -n brainactiv python=3.11
conda activate brainactiv
pip install -e .
```

### Checkpoints
Please place these in a new folder `checkpoints/` inside this repo:
- `ip-adapter_sd15.bin`, from [IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/models)
- `rgb2normal_consistency.pth`, the surface normal model from XTConsistency ([download here](https://drive.google.com/file/d/1O4G0cZDSGO0W005HALaRav5CVoii3Rx5/view?usp=sharing))
- `projection_set.npy`, CLIP embeddings of 400k natural images used to project modulation embeddings ([download here](https://drive.google.com/file/d/1t0ztK9l0zqsc2N5qKYShcbzk4o59Piiv/view?usp=sharing))
- `category_vectors/`, CLIP embeddings of 16 object and scene categories (Appendix A.1 of the paper) ([download here](https://drive.google.com/drive/folders/1wnAl2F-y4iaoNO3CtK_ht9QQKMNj2Whi?usp=sharing))

**Note**: The last two can also be computed yourself. The projection set uses images from the gated [relaion2B-en-research-safe](https://huggingface.co/datasets/laion/relaion2B-en-research-safe) dataset, so accept its terms on Hugging Face and log in first. Progress is saved every 10k images; re-run the same command to resume.
```bash
hf auth login
python scripts/compute_projection_set.py --out checkpoints/projection_set.npy
python scripts/compute_category_vectors.py --out checkpoints/category_vectors --cache checkpoints/wordnet
```

### Brain encoders (BERG package)
The experiments predict each region's response to the original and manipulated images with two encoding models from [BERG](https://github.com/gifale95/BERG): fwRF (`fmri-nsd-fwrf`, NSD volume space) and a LoRA-finetuned DINOv2 (`fmri-nsd_fsaverage-huze`, fsaverage surface). Their weights are downloaded to `checkpoints/berg/` on first use (~2.2 GB per subject for DINOv2). Note that these are not the original encoders used in the paper, but they are equally non-CLIP-based.

### Download NSD

**Note**: We define our own custom [NSD class](brainactiv/datasets/nsd.py). If you have your own NSD implementation, you can modify the experiment scripts directly to use your fMRI matrices (in the section highlighted with comments).

To download:
```bash
python scripts/download_nsd_fmri.py --out ~/Documents/Datasets/NSD       # betas + ROIs, all subjects (~30 GB temp per subject)
python scripts/download_nsd_images.py --out ~/Documents/Datasets/NSD     # 73k images (~20 GB temp)
python scripts/setup_subset_dicts.py --out ~/Documents/Datasets/NSD      # COCO categories for image subsets
python scripts/compute_clip_features.py --root ~/Documents/Datasets/NSD  # CLIP embeddings (after the two above)
```
Set `nsd.root` in the experiment configs to the folder you pass as `--out`.


### Run experiments
Each experiment takes a YAML config (see `configs/` and `brainactiv/config.py`); unknown keys raise an error. Experiments work with folders of images; you can create them from NSD image subsets (Appendix A.3 of the paper) with the [NSD class](brainactiv/datasets/nsd.py).
```bash
# 1. Manipulate images to enhance or decrease activity in a region
python experiments/exp1_image_variation.py configs/exp1_image_variation.yaml

# 2. Quantify how image features change (categories, depth, surface normals, curvature,
#    color, brightness, entropy) between the original and manipulated images of experiment 1
python experiments/exp2_feature_quantification.py configs/exp2_feature_quantification.yaml

# 3. Manipulate images to accentuate one region over another
python experiments/exp3_roi_differences.py configs/exp3_roi_differences.yaml
```
Each experiment saves its outputs (images, predicted region responses, feature differences) to the output folder in its config and prints a summary of the mean changes.

## Citation

If you find our work useful, please cite:

```
@inproceedings{garciacerdas2025brainactiv,
  title     = {{BrainACTIV}: Identifying visuo-semantic properties driving cortical selectivity using diffusion-based image manipulation},
  author    = {Garcia Cerdas, Diego and Sartzetaki, Christina and Petersen, Magnus and Roig, Gemma and Mettes, Pascal and Groen, Iris},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=CGON8Btleu}
}
```
