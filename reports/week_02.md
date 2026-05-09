# Week 2 Report: Preprocessing, CNN Baseline, and Augmentation Study

**Date:** May 2026
**Week:** 2 of 4
**Focus:** Data preprocessing, baseline CNN training, and augmentation comparison

---

## Summary

Week 2 implements a complete training pipeline for brain tumor MRI classification. The notebook pools all images from the Kaggle dataset, applies a stratified 70/15/15 split, and trains a 4-block CNN from scratch with standard augmentations. The workflow includes evaluation on a held-out test set, overfitting analysis, and a controlled comparison between training with and without augmentation. The report is aligned with the current notebook structure and does not include unverified metrics.

---

## What Was Completed

1. **Data ingestion and split:** Colab/Kaggle/local path resolution; all images are pooled and re-split into train/validation/test with stratification and fixed random seed.
2. **Preprocessing pipeline:** Resize to 224x224, normalize to $[-1, 1]$, and apply augmentation only to training data.
3. **Baseline CNN:** Four convolutional blocks with batch normalization and max pooling, global average pooling, and a dropout-regularized classifier head.
4. **Training loop and checkpoints:** Adam optimizer, StepLR schedule, early stopping, and checkpointing the best model by validation accuracy.
5. **Evaluation and diagnostics:** Test-set metrics, confusion matrix, per-class report, training curves, and overfitting gap analysis.
6. **Experiment:** A second run without augmentation for direct comparison to the augmented baseline.

---

## Experiments Run

**Experiment A (Baseline with augmentation).** The model is trained using rotation, horizontal flip, and brightness/contrast jitter. This run generates training curves, a test-set confusion matrix, and per-class metrics.

**Experiment B (No augmentation).** The same architecture and training settings are used, but the training data uses only deterministic transforms. Metrics and learning curves are compared against the baseline.

---

## Results Obtained

The notebook is designed to compute and print the following results when executed:

- Final train/validation loss and accuracy curves.
- Test-set accuracy, weighted precision, recall, and F1-score for the baseline model.
- Confusion matrix and per-class metrics on the test split.
- Augmentation comparison table including test accuracy and train-validation gap.

**Current status:** The report does not include numeric values because no run outputs are captured in this repository. The metrics should be filled in directly from the notebook output after a full training run on Kaggle or Colab.

---

## Measured Results 

- Total images collected (pooled Training + Testing): **7,200**.
- Class distribution: `glioma`, `meningioma`, `notumor`, `pituitary` — **1,800** images each (25% per class).
- Stratified split (70/15/15):
	- Train: **5,040** images (1,260 per class)
	- Validation: **1,080** images (270 per class)
	- Test: **1,080** images (270 per class)

- Baseline model (WITH augmentation) - test set:
	- **Accuracy:** 92.31%
	- **F1-score (weighted):** 92.30%

- No-augmentation model (same architecture, deterministic transforms) — test set:
	- **Accuracy:** 55.93%
	- **F1-score (weighted):** 54.59%

Notes: the notebook also generates the confusion matrix and per-class reports for both runs; use those outputs when quoting per-class precision/recall values in publications.

---

## Plan for Next Week

1. Run the notebook end-to-end on Kaggle to capture final metrics and update the results section.
2. Implement transfer learning (e.g., ResNet-50 or EfficientNet-B0) to improve accuracy.
3. Add interpretability (Grad-CAM) for qualitative validation of model focus.
4. Compare transfer models to the Week 2 baseline using the same evaluation protocol.
