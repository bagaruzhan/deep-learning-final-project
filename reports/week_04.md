# Week 4 Report: Final Improvements, Comparison, and Deployment

**Project:** Brain Tumor MRI Classification  
**Week:** 4 of 4  
**Date:** May 2026  

## 1. What Was Done in Week 4

Week 4 started from the Week 3 transfer-learning pipeline and focused on three goals:

1. Improve robustness of the best ResNet18 setup.  
2. Re-evaluate all project models on one fixed held-out test split.  
3. Prepare deployment artifacts for the Streamlit app.  

Implemented in notebook code:

- stronger train augmentations (RandomGrayscale, GaussianBlur, stronger contrast jitter)
- improved ResNet18 classifier head with BatchNorm
- CosineAnnealingLR scheduler
- label smoothing (CrossEntropyLoss with label_smoothing=0.1)
- unified evaluation of all models with accuracy, precision, recall, F1
- saving deployment checkpoint as results/models/best_model_deploy.pth

## 2. Improvements vs Week 3

Main differences from Week 3:

- data pipeline is more robust to visual variation in MRI quality
- classifier head has higher capacity and normalization
- optimization schedule is smoother over epochs
- confidence calibration is improved by label smoothing

These changes were intended to improve generalization and reduce overconfident mistakes.

## 3. Model Comparison (Week 4 Notebook Outputs)

All metrics below are the Week 4 notebook final test results.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet18 Week 4 (Improved) | 98.33% | 98.34% | 98.33% | 98.34% |
| EfficientNet-B0 (W3 Exp 3) | 98.24% | 98.24% | 98.24% | 98.24% |
| ResNet18 FineTune (W3 Exp 2) | 98.06% | 98.07% | 98.06% | 98.06% |
| Custom CNN (Week 2) | 86.85% | 87.96% | 86.85% | 86.84% |
| ResNet18 Frozen (W3 Exp 1) | 85.46% | 85.23% | 85.46% | 84.94% |

Best model in Week 4 notebook: ResNet18 Week 4 (Improved).

## 4. Final Evaluation Details

Evaluation protocol:

- fixed stratified test split
- same split for all models
- confusion matrix and per-class report generated for best model
- error visualization includes correct and incorrect samples plus confidence
- Grad-CAM used for qualitative interpretability checks

This keeps comparison fair and supports reproducibility.

## 5. Problems Encountered and Solutions

Problem 1: metrics from old reports did not match Week 4 notebook outputs.  
Solution: synchronized Week 4 documentation with actual notebook output values.

Problem 2: model confidence could be over-optimistic on difficult samples.  
Solution: label smoothing added in training and confidence shown explicitly in app.

Problem 3: deployment reliability required a stable checkpoint path.  
Solution: best model exported to a fixed deploy filename for app loading.

## 6. Week 3 vs Week 4 Interpretation

Week 3 and Week 4 are not directly interchangeable because Week 4 re-ran the full benchmark with a modified training recipe and checkpoint set. The strongest Week 3 single-run result (99.07%) came from its own experiment conditions. In Week 4, all models were compared under one unified final pipeline and yielded a different ranking scale centered around 98%.

## 7. If Extended Beyond Week 4

Planned extensions:

1. multi-seed experiments with confidence intervals for statistical stability  
2. probability calibration (temperature scaling) and reliability diagrams  
3. harder-case mining for glioma vs meningioma confusion  
4. external validation on additional MRI sources to test domain shift  
5. optional 3D or multi-slice modeling for anatomical context  

## 8. Conclusion

Week 4 delivered a complete end-to-end final stage: improved training recipe, full benchmark, and deployable app checkpoint. The final best model in Week 4 is ResNet18 Week 4 (Improved), and the project now includes reproducible evaluation and deployment artifacts aligned with implementation.