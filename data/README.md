# Dataset Guide - Brain Tumor MRI

## Overview

This directory contains the medical imaging dataset used for brain tumor classification. The dataset consists of MRI brain scans classified into four categories.

## Dataset Source

**Name:** Brain Tumor MRI Dataset  
**Source:** [Kaggle Datasets](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
**Size:** ~7,000 images (~2.5GB)  
**License:** Kaggle Dataset License  

## Classes

| Class | Description | Size | Type |
|-------|-------------|------|------|
| **glioma** | Gliomas (tumors of glial cells) | ~1,426 images | Malignant |
| **meningioma** | Meningiomas (tumors of brain membranes) | ~1,376 images | Benign/Malignant |
| **notumor** | Healthy brain scans (no tumor) | ~1,595 images | Negative |
| **pituitary** | Pituitary gland tumors | ~1,457 images | Benign/Malignant |

**Total:** ~5,854 images (balanced across classes)

## Image Specifications

- **Resolution:** 512 × 512 pixels
- **Format:** JPEG (.jpg)
- **Modality:** T2-weighted MRI (brain axial view)
- **Color Space:** Grayscale (3-channel RGB, uniform channels)
- **Intensity Range:** 0-255

## Directory Structure

```
data/
├── README.md (this file)
└── raw/
    ├── Training/ (70% of data - training set)
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    ├── Testing/ (15% of data - test set)
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Validation/ (15% of data - validation set)
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

**Note:** Dataset folder structure may vary; notebooks handle multiple formats.

## Download Instructions

### Option 1: Kaggle API (Recommended)

1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Download API token from [Kaggle Account Settings](https://www.kaggle.com/settings/account)
   - Click "Create New Token"
   - Save `kaggle.json` to `~/.kaggle/` (or `C:\Users\<Username>\.kaggle\` on Windows)

3. Download dataset:
   ```bash
   cd data/
   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
   unzip brain-tumor-mri-dataset.zip
   ```

### Option 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click "Download" button (requires Kaggle login)
3. Extract ZIP to `data/raw/`

### Option 3: Alternative Source

If Kaggle access is unavailable, the dataset is also hosted at:
- [GitHub Alternative](https://github.com/topics/brain-tumor-detection)
- [Zenodo Repository](https://zenodo.org/)

## Data Imbalance Analysis

✅ **Dataset is relatively balanced:**
- Smallest class: ~1,376 images (meningioma)
- Largest class: ~1,595 images (notumor)
- Imbalance ratio: 1.16:1 (acceptable)

**Recommendation:** Class weights in training loss are optional but beneficial

## Quality Notes

### Known Characteristics
- ✅ Clean, pre-processed MRI scans
- ✅ Consistent image dimensions (512×512)
- ✅ Professional medical imaging database
- ✅ Representative of real clinical data

### Preprocessing Already Done
- Images are skull-stripped
- Anatomically aligned
- Artifact removal performed

### No Preprocessing Needed For
- Alignment (already done)
- Skull stripping (already done)
- Intensity normalization (applied in pipeline)

## Data Loading Example

```python
from PIL import Image
import torch
from torchvision import transforms

# Load single image
img_path = 'data/raw/Training/glioma/image_001.jpg'
img = Image.open(img_path)
img.show()

# Image statistics
import numpy as np
img_array = np.array(img)
print(f"Shape: {img_array.shape}")
print(f"Min: {img_array.min()}, Max: {img_array.max()}")
print(f"Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
```

## Storage & Performance Tips

### Disk Space
- Raw dataset: ~2.5GB
- After preprocessing: ~2.5GB (same)
- With augmentation cache: ~5-10GB
- **Recommendation:** SSD recommended for faster loading

### Memory Optimization
- Use data loaders with `num_workers > 0`
- Implement on-the-fly augmentation
- Don't load entire dataset into RAM
- Use mixed precision (fp16) for large batches

### Loading Speed
Expected loading times (first run):
- Single image: <1ms
- Batch of 32: <50ms
- Full train set (DataLoader): ~30 seconds per epoch

## Data Privacy & Ethics

⚠️ **Important Considerations:**

1. **Medical Data:** Patient-anonymized but represents real clinical cases
2. **No Re-identification:** Do not attempt to identify patients
3. **Ethical Use:** Only for educational/research purposes
4. **License Compliance:** Follow Kaggle dataset license
5. **GitHub Policy:** 
   - ❌ DO NOT upload raw images to GitHub
   - ✅ DO upload code, notebooks, and results
   - ✅ Add `data/raw/` to `.gitignore`

## File Size Breakdown

```
Training Set:   ~1.8GB (60%)
Validation Set: ~0.3GB (15%)
Testing Set:    ~0.4GB (15%)
```

## Troubleshooting

### Images not loading?
- Verify JPEG files are not corrupted: `identify -verbose image.jpg`
- Check file permissions
- Ensure path separators match OS (Windows uses `\`, Linux/Mac use `/`)

### Memory issues?
- Reduce batch size in DataLoader
- Resize images to 256×256 or 224×224
- Use `pin_memory=True` for CUDA transfers

### Class imbalance concerns?
- Current ratio is minimal
- Stratified k-fold recommended for validation
- Optional: weighted loss functions in Week 2

## Citation

If using this dataset in publication, cite:
```bibtex
@misc{brain-tumor-mri,
  author = {Nickparvar, Masoud},
  title = {Brain Tumor MRI Dataset},
  year = {2023},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}}
}
```

## Next Steps

Once dataset is downloaded and verified:
1. Run `notebooks/week_01.ipynb` to start EDA
2. Verify class distribution with provided scripts
3. Check image statistics and preprocessing needs
4. Proceed to baseline model implementation

---

**Last Updated:** April 2026  
**Verified with:** PyTorch 2.0+, PIL 10.0+  
**Status:** Ready for Week 1 EDA
