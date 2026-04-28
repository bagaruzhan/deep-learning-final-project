# Brain Tumor MRI Classification - Deep Learning Project

## Overview

This is a comprehensive deep learning project for **medical image classification** focusing on brain tumor detection and classification using MRI scans. The project implements state-of-the-art CNN architectures with transfer learning to classify brain tumors into four categories: glioma, meningioma, pituitary tumor, or no tumor.

**Project Duration:** 4 weeks 
**Domain:** Medical Imaging / Computer Vision  

##  Problem Statement

Brain tumors are a critical health concern with high mortality rates if not detected early. Manual classification of MRI scans by radiologists is time-consuming and prone to human error. This project develops an AI-powered system to:

- **Automate** brain tumor detection from MRI images
- **Classify** tumor types accurately and quickly
- **Assist** radiologists in clinical decision-making
- **Improve** diagnostic efficiency and patient outcomes

The model achieves **95%+ validation accuracy** with proper calibration and can serve as a clinical decision support system.

##  Dataset

**Source:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Statistics:**
- **Total Images:** ~7,000 MRI scans
- **Image Format:** JPEG (512×512 pixels)
- **Classes:** 4 balanced classes
  - Glioma (brain tumor)
  - Meningioma (membrane tumor)
  - Pituitary (pituitary gland tumor)
  - No Tumor (healthy control)
- **Modality:** T2-weighted MRI
- **Train/Val/Test Split:** 70/15/15 (standard)

## Project Structure

```
brain-tumor-mri-deep-learning/
├── README.md                 
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── data/
│   ├── README.md              
│   └── raw
│
├── notebooks/
│   ├── week_01.ipynb           # Week 1: EDA 
│   ├── week_02.ipynb           # Week 2: Baseline Model & CNN Architecture & Training
│   ├── week_03.ipynb           # Week 3: Transfer Learning & Fine-tuning
│   └── week_04.ipynb           # Week 4: Grad-CAM & Advanced Analysis
│
├── reports/
│   ├── week_01.md              # Week 1 summary & findings
│   ├── week_02.md              # Week 2 progress report
│   ├── week_03.md              # Week 3 transfer learning results
│   └── week_04.md              # Week 4 final report
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Dataset loading utilities
│   ├── preprocessor.py         # Image preprocessing functions
│   ├── models.py               # Model architectures
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   └── visualization.py        # Visualization utilities
│
└── results/
    ├── models/                 # Saved model weights (`.pth` or `.h5`)
    ├── plots/                  # Visualizations (confusion matrices, etc.)
    └── predictions/            # Output predictions on test set
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
   - Visit [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Download and extract to `data/raw/`
   - See `data/README.md` for detailed instructions

5. **Run Week 1 notebook**
   ```bash
   jupyter notebook notebooks/week_01.ipynb
   ```

## Weekly Breakdown

### Week 1: Exploratory Data Analysis & Baseline
-  Dataset loading and exploration
-  EDA with statistical analysis
-  Image preprocessing fundamentals

### Week 2: CNN Architecture & Training
- Deep CNN implementation
- Training pipeline with callbacks
- Hyperparameter tuning
- Learning curves analysis

### Week 3: Transfer Learning
- ResNet50 / EfficientNet implementation
- Fine-tuning strategies
- Comparative analysis
- Performance optimization

### Week 4: Advanced Analysis & Deployment
- Grad-CAM visualization
- Confusion matrix analysis
- Model interpretability
- Deployment preparation

## Technologies Used

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch / TensorFlow |
| **Image Processing** | PIL, OpenCV |
| **Data Science** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | tqdm, joblib |

##  Expected Performance

- **Week 1 EDA**
- **Week 2 CNN:** ~90-92% accuracy
- **Week 3 Transfer Learning:** ~95-97% accuracy
- **Week 4 Optimized Model:** ~97%+ accuracy

##  Important Notes

### GPU Requirements
- Training times:
  - Week 1 Baseline: 2-5 minutes (CPU acceptable)
  - Week 2 CNN: 10-20 minutes (GPU recommended)
  - Week 3 Transfer Learning: 5-15 minutes (GPU recommended)


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

##  License

This project is for educational purposes. The dataset is available under the Kaggle license.

##  Contact & Support

For questions about the project:
- Check the weekly reports in `reports/`
- Review notebook comments and documentation
- Consult the dataset README in `data/README.md`


