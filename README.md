#  Pneumonia Detection using Deep Learning

A state-of-the-art deep learning model for detecting pneumonia from chest X-ray images using **DenseNet121 + CBAM Attention + Focal Loss**.

##  Project Overview

This project develops and evaluates a binary classification model to automatically detect pneumonia in chest X-ray images. The model achieves **96.2% sensitivity** and **0.9275 AUC** on the test set, making it highly effective for medical screening applications.

### Key Features
- **Architecture**: DenseNet121 + Convolutional Block Attention Module (CBAM)
- **Loss Function**: Focal Loss (handles class imbalance)
- **Explainability**: Grad-CAM visualizations to show decision-making regions
- **Comprehensive Metrics**: ROC curves, PR curves, confusion matrices, and classification reports

##  Model Performance

| Metric | Score |
|--------|-------|
| **ROC AUC** | 0.9275 |
| **Average Precision (AP)** | 0.9318 |
| **Sensitivity (Recall)** | 96.2% |
| **Specificity** | 73.9% |
| **Precision** | 86.0% |
| **Accuracy** | 87.9% |

### Confusion Matrix
```
                  Predicted
                   Normal  Pneumonia
Actual Normal     [ 346      122]
       Pneumonia  [  30      750]
```

##  Project Structure

```
PNEUMONIA-DETECTION/
├── src/
│   ├── models/
│   │   ├── densenet_attention.py    # DenseNet121 + CBAM model
│   │   └── __init__.py
│   ├── metrics/
│   │   ├── roc_curve.py             # ROC curve generation
│   │   ├── plot_pr_curve.py         # Precision-Recall curve
│   │   ├── plot_confusion_matrix.py # Confusion matrix visualization
│   │   └── __init__.py
│   ├── explainability/
│   │   ├── gradcam.py               # Grad-CAM visualization
│   │   └── __init__.py
│   ├── train.py                      # Training script with 2-phase training
│   ├── evaluate.py                   # Model evaluation
│   ├── data_loader.py               # Data loading & preprocessing
│   ├── dataset.py                    # Custom dataset class
│   ├── augmentations.py             # Data augmentation
│   ├── losses.py                    # Custom loss functions (Focal Loss)
│   └── config.py                    # Configuration settings
├── notebooks/                        # Jupyter notebooks for exploration
├── data/
│   ├── raw/                         # Raw dataset (not tracked)
│   └── processed/                   # Processed data
├── models/                          # Trained model checkpoints
│   ├── phase1_best.pth
│   └── phase2_best.pth
├── outputs/                         # Generated visualizations
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── gradcam_result.png
├── test_loader/                     # Data loading tests
├── requirement.txt                  # Project dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ftkhirabditanaya/PNEUMONIA-DETECTION.git
cd PNEUMONIA-DETECTION
```

### 2. Set up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

### 4. Download and Prepare Dataset
- Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle
- Extract to `data/raw/chest_xray/`

### 5. Train the Model
```bash
$env:PYTHONPATH = "."; python src/train.py
```

### 6. Evaluate & Generate Metrics
```bash
# ROC Curve
python src/metrics/roc_curve.py

# Precision-Recall Curve
python src/metrics/plot_pr_curve.py

# Confusion Matrix
python src/metrics/plot_confusion_matrix.py

# Grad-CAM Visualization
python src/explainability/gradcam.py
```

##  Training Strategy

The model uses **2-phase training**:

**Phase 1: Classifier Training (Backbone Frozen)**
- Train only CBAM + classifier layers
- Epochs: 25
- Learning Rate: 1e-4

**Phase 2: Fine-tuning (Full Network)**
- Train entire network with frozen early layers
- Epochs: 10
- Learning Rate: 1e-5
- Lower learning rate to preserve pre-trained features

##  Explainability

### Grad-CAM Visualization
Run to see which regions of X-rays the model focuses on:
```bash
python src/explainability/gradcam.py
```

Sample output shows attention heatmaps overlaid on the original X-ray image, revealing regions contributing to pneumonia detection decisions.

##  Technologies Used

- **Deep Learning**: PyTorch 2.0+
- **Pre-trained Models**: TorchVision
- **Data Processing**: OpenCV, PIL, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn
- **Image Augmentation**: Albumentations

##  Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration, optional)
- Kaggle API (for downloading dataset)

See `requirement.txt` for complete list.

##  Dataset Information

**Chest X-Ray Images (Pneumonia) Dataset**
- **Total Images**: 5,856
- **Classes**: 2 (Normal, Pneumonia)
- **Split**:
  - Training: 8,371 images (2.88x imbalance)
  - Validation: 2,093 images
  - Testing: 1,248 images
  - Prediction :0  (Normal Lungs)
  - Prediction : 1 (Pneumonia Affected Lungs)

##  Medical Disclaimer

This project is for **educational and research purposes only**. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decision-making.

##  Contributing

Contributions are welcome! Feel free to:
- Submit pull requests
- Report bugs
- Suggest improvements
- Add new features

##  Acknowledgments

- Dataset: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- CBAM: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- Focal Loss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

##  Contact

For questions or inquiries, feel free to reach out via GitHub issues.

---

** If you found this project useful, please consider giving it a star!**
