# Brain Tumor MRI Classification - Deep Learning Project

- Student: Aruzhan Bagytzhan
- Project Title: Brain Tumor MRI Classification
- 100


## Overview

This is a comprehensive deep learning project for **medical image classification** focusing on brain tumor detection and classification using MRI scans. The project implements state-of-the-art CNN architectures with transfer learning to classify brain tumors into four categories: glioma, meningioma, pituitary tumor, or no tumor.

- **Project Duration:** 4 weeks 
- **Domain:** Medical Imaging / Computer Vision
- Task Type: Image Classification (Deep Learning)

##  Problem Statement

Brain tumors are a critical health concern with high mortality rates if not detected early. Manual classification of MRI scans by radiologists is time-consuming and prone to human error.
This project aims to:

- **Automate** brain tumor detection from MRI scans
- **Classify** tumor types into four categories
- **Support** medical image analysis workflows
- **Improve** consistency of predictions using deep learning

##  Dataset

**Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle

**Statistics:**
- **Total Images:** 7,200 MRI scans
- **Image Format:** JPEG (512×512 pixels)
- **Classes:** 4 balanced classes (1,800 images each)
  - Glioma (brain tumor)
  - Meningioma (membrane tumor)
  - Pituitary (pituitary gland tumor)
  - No Tumor (healthy control)
- **Modality:** T2-weighted MRI
- **Train/Val/Test Split:** 70/15/15
  - Training: 5,040 images
  - Validation: 1,080 images
  - Test: 1,080 images

## Project Structure

```
brain-tumor-mri-deep-learning/
├── README.md                 
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── app.py                       # Streamlit 
│
├── data/
│   ├── README.md              
│   └── raw
│
├── notebooks/
│   ├── week_01.ipynb           # Week 1: EDA 
│   ├── week_02.ipynb           # Week 2: Baseline Model & CNN Architecture & Training
│   ├── week_03.ipynb           # Week 3: Transfer Learning & DL models
│   └── week_04.ipynb           # Week 4: Grad-CAM & Advanced Analysis
│
├── reports/
│   ├── week_01.md              # Week 1 summary & findings
│   ├── week_02.md              # Week 2 progress report
│   ├── week_03.md              # Week 3 dl models result
|   ├── week_04.md              # Week 4 final report
│   └── final-report.md         # final report
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Dataset loading utilities
│   ├── models.py               # Model architectures
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation metrics
```

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for training)
- ~10GB storage for dataset

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/brain-tumor-mri-deep-learning.git
   cd brain-tumor-mri-deep-learning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Visit [the Kaggle dataset page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Download and extract to `data/raw/`
   - See `data/README.md` for detailed instructions

5. **If you run in Colab**
   - Mount Google Drive and place the dataset under `data/raw/`
   - The Week 2 notebook auto-detects local and Drive paths
   - You can also use the Kaggle API if your token is configured in Colab

6. **Run Week 1 notebook**
   ```bash
   jupyter notebook notebooks/week_01.ipynb
   ```

## Run Streamlit App

   ```bash
   streamlit run app.py
   ```

Features:
- Image upload
- Model selection
- Prediction probabilities
- Grad-CAM visualization

## Weekly Breakdown

### Week 1: Exploratory Data Analysis & Baseline
-  Dataset loading and exploration
-  EDA with statistical analysis
-  Image preprocessing fundamentals

### Week 2: CNN Architecture & Training
- 4-layer custom CNN with BatchNorm
- Strong data augmentation

### Week 3: Transfer Learning
- ResNet18 / EfficientNet-B0
- Frozen vs fine-tuning comparison
- Optimization strategies (LR scheduling, AdamW)

### Week 4: Advanced Analysis & Deployment
- Grad-CAM interpretability
- Model comparison
- Final optimization
- Streamlit deployment preparation


# Models
## Custom CNN (Week 2)

Architecture:

- 4 convolutional layers
- Batch normalization
- ReLU activation
- Max pooling
- Adaptive average pooling
- CrossEntropyLoss
- Adam optimizer

Results:

| Setup | Accuracy             | F1-score |
| ---- | --------------------- | ------------ | 
| Without augmentation | 55.93% | 54.59%        | 
| With augmentation | 92.31%      | 92.30%        |


## Transfer Learning (Week 3)

Models tested:
- ResNet18
- EfficientNet-B0
- ResNet18 full fine-tuning

Training techniques:

Full fine-tuning
Frozen backbone comparison
Differential learning rates
Adam / AdamW optimizers
Learning rate scheduling
Early stopping
Final Results

| Experiment | Model | Strategy | Validation Accuracy | Validation F1 |
|------------|-------|----------|---------------------|---------------|
| 1 | ResNet18 | Frozen backbone, trained head only | 81.48% | 81.01% |
| 2 | ResNet18 | Full fine-tuning with differential learning rates | **98.24%** | **98.24%** |
| 3 | EfficientNet-B0 | Full fine-tuning with AdamW and cosine LR | 97.78% | 97.78% |

The best validation model, **ResNet18 full fine-tuning**, was then evaluated on the held-out test split:

| Metric | Test Result |
|--------|-------------|
| Accuracy | **99.07%** |
| Precision | **99.07%** |
| Recall | **99.07%** |
| F1-score | **99.07%**|

## Final Results (Test Set):

| Rank | Model | Test Accuracy | Precision | Recall | F1-Score |
|------|-------|--------------|-----------|--------|----------|
| 1 | ResNet18 Week 4 (Best) | **98.33%** | **98.34%** | **98.33%** | **98.34%** |
| 2 | EfficientNet-B0 (W3 Exp 3) | 98.24% | 98.24% | 98.24% | 98.24% |
| 3 | ResNet18 FineTune (W3 Exp 2) | 98.06% | 98.07% | 98.06% | 98.06% |
| 4 | Custom CNN (Week 2) | 86.85% | 87.96% | 86.85% | 86.84% |
| 5 | ResNet18 Frozen (W3 Exp 1) | 85.46% | 85.23% | 85.46% | 84.94% |

## Note on Results Variation

Week 3 and Week 4 results differ due to:
- unified re-evaluation pipeline in Week 4
- changes in augmentation and training recipe
- different checkpoint selection strategy (best-validation vs final benchmark run)
Week 4 focuses on reproducibility and fair comparison across all models.

## Key Findings
- Transfer learning significantly outperforms CNN from scratch
- Full fine-tuning works better than frozen features
- ResNet18 slightly outperformed EfficientNet-B0
- Most errors occur between glioma and meningioma
- Pituitary tumors are easiest to detect

## Technologies Used

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch / TensorFlow |
| **Image Processing** | PIL, OpenCV |
| **Data Science** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | tqdm, joblib |

##  Important Notes

### Data Storage
- **Raw dataset NOT included in repository** — download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place in `data/raw/`
- Configured in `.gitignore` to prevent accidental commits of large files
- Model weights (`.pth` files) in `results/models/` are also ignored

### Environment Compatibility
- Notebook auto-detects local, Colab, and Kaggle paths
- Uses Kaggle API for automated downloads if `kaggle.json` is configured
- CPU/GPU handling is automatic (sets `pin_memory=True` only on GPU systems)
- Training times (estimated):
  - Week 1 EDA: ~5 minutes (CPU acceptable)
  - Week 2 CNN: 15-25 minutes (GPU recommended)
  - Week 3 Transfer Learning: 1 hour (GPU recommended)
  - Week 4 Optimization: 2 hours (GPU recommended)

**Key Concepts:**
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Medical Image Analysis
- Deep Learning Best Practices
- Model Interpretability (Grad-CAM)

##  Project Guidelines

### Code Style
- PEP 8 compliant
- Descriptive variable names
- Comprehensive docstrings
- Type hints where applicable

### Commits
Commit after each major milestone:
```
git add .
git commit -m "Week 1: EDA and baseline model"
```

### Reproducibility
- Set random seeds (NumPy, PyTorch)
- Document all hyperparameters
- Save training logs and metrics

##  Contributing

This is an educational project. For feedback or improvements:
1. Create an issue describing the suggestion
2. Submit a pull request with changes
3. Ensure code follows project guidelines

## Conclusion

This project demonstrates the effectiveness of transfer learning for medical image classification. ResNet18 with fine-tuning achieved the best overall performance.

## Limitations

- 2D slice-based MRI analysis (no volumetric context)
- Dataset from single source (Kaggle)
- No external clinical validation
- Potential overfitting to dataset distribution

## Future Work

- 3D CNN for volumetric MRI analysis
- External dataset validation
- Model calibration and uncertainty estimation
- Deployment with clinical-grade UI improvements


