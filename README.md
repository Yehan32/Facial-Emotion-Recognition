# Facial-Emotion-Recognition

## Overview
This project implements a facial emotion recognition system using traditional machine learning techniques. The system classifies six basic emotions (anger, fear, happiness, sadness, surprise, and neutral) from facial images.

## Methodology
- **Feature Extraction:** Histogram of Oriented Gradients (HOG)
- **Classification:** Support Vector Machine (SVM)

## Files
1. **my_preprocessing.py** - Dataset preprocessing and data augmentation
2. **extract_features.py** - HOG feature extraction
3. **train_svm_model.py** - SVM training, hyperparameter tuning, and evaluation

## Requirements
- Python 3.7 or higher
- Required libraries:
  - opencv-python (cv2)
  - pillow (PIL)
  - numpy
  - scikit-learn
  - scikit-image
  - matplotlib
  - seaborn

## Installation

Install all required libraries using pip:
```
pip install opencv-python
pip install pillow
pip install numpy
pip install scikit-learn
pip install scikit-image
pip install matplotlib
pip install seaborn
```

### How to run
1. Extract datasets to project directory.
2. Run the preprocessing script to prepare the datasets.
3. Extract HOG features from the preprocessed images.
4. Run the model trainig.

## Output Files
After running all scripts, you will have:
- Trained models: `CK_svm_model.pkl`, `JAFFE_svm_model.pkl`
- Confusion matrices: `CK_confusion_matrix.png`, `JAFFE_confusion_matrix.png`
- Performance metrics: `CK_metrics.png`, `JAFFE_metrics.png`
- Comparison chart: `dataset_comparison.png`
- Classification reports: `CK_classification_report.txt`, `JAFFE_classification_report.txt`

## Dataset Structure

Required folder structure:
```
project/
├── dataset/
│   ├── train/
│   └── test/
├── dataset
│   ├── train/
│   └── test/
├── my_preprocessing.py
├── extract_features.py
└── train_svm_model.py
```

## License
This project is for educational purposes only.

## Author
Yehan Heenpella - UCLan Student  
Module: CO3519 Artificial Intelligence

