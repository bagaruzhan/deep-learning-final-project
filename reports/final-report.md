# Final Report: Brain Tumor MRI Classification Using Deep Learning

**Course:** Applied Deep Learning  
**Period:** 4 weeks  
**Date:** May 2026  

## 1. Project Title

Brain Tumor MRI Classification Using CNN and Transfer Learning Models

## 2. Problem Statement

The task is to classify brain MRI images into four categories: glioma, meningioma, notumor, and pituitary. The objective is to build a reproducible pipeline that demonstrates strong deep-learning performance and transparent evaluation.

This is an educational project and not a clinical diagnostic tool.

## 3. Dataset Description

Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset summary used in notebooks:

- total images: 7,200  
- classes: 4  
- per class: 1,800  
- split: stratified 70/15/15 with SEED=42  

Split counts:

- train: 5,040  
- validation: 1,080  
- test: 1,080  

Repository policy: dataset files are not included in Git and must be downloaded separately.

## 4. Preprocessing

Common preprocessing:

- RGB conversion and resize to 224x224  
- ImageNet normalization (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225])  

Week 4 training augmentation includes:

- Resize to 244 then RandomCrop 224  
- RandomHorizontalFlip  
- RandomRotation 15 degrees  
- ColorJitter (brightness 0.3, contrast 0.4, saturation 0.1)  
- RandomAffine (shear 10)  
- RandomGrayscale (p=0.1)  
- GaussianBlur (kernel 3, sigma 0.1 to 1.0)  

Validation/test transform is deterministic:

- Resize 224  
- ToTensor  
- Normalize  

## 5. Model Architectures

Compared models:

1. Custom CNN (from scratch, Week 2)  
2. ResNet18 frozen backbone (Week 3 experiment 1)  
3. ResNet18 full fine-tuning (Week 3 experiment 2)  
4. EfficientNet-B0 full fine-tuning (Week 3 experiment 3)  
5. ResNet18 Week 4 improved variant (BatchNorm head + training improvements)  

Week 4 ResNet18 head:

- BatchNorm1d(512)  
- Dropout(0.5)  
- Linear(512, 512)  
- BatchNorm1d(512)  
- ReLU  
- Dropout(0.3)  
- Linear(512, 4)  

## 6. Training Setup

Main setup in transfer-learning stages:

- batch size: 32  
- max epochs: 30  
- early stopping patience: 7  
- optimizer: Adam/AdamW depending on experiment  
- differential LR where applicable (backbone lower than head)  
- Week 4 scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-6)  
- Week 4 loss: CrossEntropyLoss(label_smoothing=0.1)  

Evaluation protocol:

- validation used for model selection  
- test used for final reporting  
- metrics: accuracy, precision, recall, F1  
- confusion matrix and per-class report for best model  

## 7. Experiments and Comparison

### 7.1 Week 3 reference result

Best Week 3 model (ResNet18 full fine-tuning) in Week 3 notebook:

- test accuracy: 99.07%  
- precision: 99.07%  
- recall: 99.07%  
- F1: 99.07%  

### 7.2 Week 4 final benchmark

Week 4 notebook re-ran a unified comparison and produced the following test metrics:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet18 Week 4 (Improved) | 98.33% | 98.34% | 98.33% | 98.34% |
| EfficientNet-B0 (W3 Exp 3) | 98.24% | 98.24% | 98.24% | 98.24% |
| ResNet18 FineTune (W3 Exp 2) | 98.06% | 98.07% | 98.06% | 98.06% |
| Custom CNN (Week 2) | 86.85% | 87.96% | 86.85% | 86.84% |
| ResNet18 Frozen (W3 Exp 1) | 85.46% | 85.23% | 85.46% | 84.94% |

Week 4 best model: ResNet18 Week 4 (Improved).

## 8. Final Results and Confusion Matrix Interpretation

Final selected model (Week 4):

- accuracy: 98.33%  
- precision: 98.34%  
- recall: 98.33%  
- F1: 98.34%  

Confusion-matrix interpretation:

- strongest diagonal concentration indicates robust overall classification  
- most residual confusion occurs between glioma and meningioma  
- pituitary and notumor classes are generally easier due to stronger visual/anatomical cues  

## 9. Error Analysis

Observed error patterns are consistent across weeks:

1. Glioma vs meningioma overlap:  
These classes can have similar bright/irregular morphology in 2D slices. Without full 3D anatomical context, the boundary remains difficult.

2. Confidence on hard cases:  
Even with label smoothing, some wrong predictions remain moderately high confidence. This motivates post-hoc calibration.

3. Information loss from resizing/cropping:  
Converting all scans to 224x224 can remove subtle lesion details, especially near borders.

4. Domain-shift risk:  
Single-source Kaggle data may not represent scanner/protocol variation from real hospitals.

5. Class-specific difficulty asymmetry:  
Pituitary often has stronger location cues; glioma/meningioma remains the hardest pair.

## 10. Limitations

- single public dataset source  
- 2D image classification instead of full 3D MRI sequence analysis  
- no external validation cohort  
- confidence calibration not fully optimized  
- educational scope only; no clinical certification  

## 11. Conclusion

The project achieved a complete and reproducible deep-learning workflow from baseline CNN to transfer learning and deployment. Key conclusions:

1. Transfer learning with full fine-tuning provides major gains over frozen-backbone baselines.  
2. Training recipe improvements in Week 4 (augmentation, optimization, regularization) improve robustness and maintain top performance in unified comparison.  
3. Evaluation beyond accuracy (precision, recall, F1, confusion matrix, error analysis) gives a stronger and more defensible result.  
4. A working Streamlit app demonstrates practical deployment readiness for course requirements.  

## 12. References

1. Brain Tumor MRI Dataset (Kaggle): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset