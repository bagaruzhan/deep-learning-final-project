# Week 3 Report - Transfer Learning Experiments

**Project:** Brain Tumor MRI Classification  
**Dataset:** Brain Tumor MRI Dataset (Kaggle)  
**Week:** 3

## What Was Done

Week 3 focused on transfer learning using ImageNet-pretrained CNNs. The notebook builds a reproducible 70/15/15 stratified split from the labelled local dataset copy (7,200 images, 1,800 per class). The training split is used for weight updates, the validation split is used for model selection and early stopping, and the held-out test split is used only after the best model is selected.

The notebook includes GPU/CPU device handling, ImageNet normalization, data augmentation, reusable training and validation functions, checkpointing, early stopping, weighted precision/recall/F1 metrics, confusion matrix, error analysis, Grad-CAM, and checkpoint reload verification.

## Experiments and Results

| Model | Strategy | Selection Set | Accuracy | Precision | Recall | F1 |
|-------|----------|---------------|----------|-----------|--------|----|
| Week 2 Custom CNN | Baseline reference | Test | 92.31% | 92.30% | 92.31% | 92.30% |
| ResNet18 Frozen | Train classifier head only | Validation | 81.48% | 81.39% | 81.48% | 81.01% |
| ResNet18 Fine-Tuned | Full fine-tuning, differential LRs | Validation | **98.24%** | **98.25%** | **98.24%** | **98.24%** |
| EfficientNet-B0 | Full fine-tuning, AdamW + cosine LR | Validation | 97.78% | 97.81% | 97.78% | 97.78% |

The selected model was **ResNet18 full fine-tuning** because it had the highest validation accuracy. Its final held-out test results were:

| Metric | Test Result |
|--------|-------------|
| Accuracy | **99.07%** |
| Precision | **99.07%** |
| Recall | **99.07%** |
| F1-score | **99.07%** |

## Key Findings

Full fine-tuning clearly outperformed frozen feature extraction. The frozen ResNet18 was fast, but the MRI task needed the backbone to adapt beyond generic ImageNet features. EfficientNet-B0 was competitive, but ResNet18 fine-tuning performed slightly better on this split and became the Week 3 best model.

The model made 10 mistakes on 1,080 test images. The main error pattern was confusion between **glioma** and **meningioma**, which is realistic because these tumor classes can look visually similar after resizing. Pituitary had the strongest recall (99.63%), likely because its anatomical location is distinctive. Grad-CAM was added to check whether predictions focus on meaningful image regions rather than background artifacts.

## Next Steps

Week 4 should focus on calibration and robustness: temperature scaling, Grad-CAM review of all misclassified images, possible skull-stripping preprocessing, and exporting the best ResNet18 checkpoint for a simple demo or deployment workflow.
