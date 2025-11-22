# Credit Card Fraud Detection using CNN and SMOTE

## Project Overview

This project implements a ***1D Convolutional Neural Network (CNN)*** to detect fraudulent credit card transactions. A major focus of this study is addressing the severe class imbalance (only 0.17% fraud) using ***SMOTE (Synthetic Minority Oversampling Technique)***.

The project compares model performance on two datasets:

1. Imbalanced Data: Original dataset.
2. Balanced Data: Oversampled using SMOTE.

## Dataset

- Source: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Size: 284,807 transactions
- Distribution: 99.83% Legitimate, 0.17% Fraudulent
- Features: 28 PCA-transformed features (V1-V28) + Time & Amount.

## Methodology

* Preprocessing: `StandardScaler` applied to 'Time' and 'Amount'.

* Sampling: Stratified split (80% Train / 20% Test). SMOTE applied only to the training set to prevent data leakage.

* Model Architecture: 1D CNN optimized for tabular data patterns.

## Results

The study found that while SMOTE significantly improves the detection of fraud cases (Recall), it comes at the cost of precision (more false positives).

| Metric | Without SMOTE | With SMOTE |
| --- | --- | --- |
| Accuracy | 100% | ~99% |
| Fraud Recall | 0.81 | 0.90 |
| Fraud Precision | 0.82 | 0.66 |
| Fraud F1-Score | 0.81 | 0.76|

Key Finding: Balancing the dataset using SMOTE enhances the model's sensitivity (Recall 0.90), which is critical in fraud detection to minimize missed fraudulent transactions.

## How to Run

1. Clone the repo:
```
git clone https://github.com/ashen-pabasara/CNN-credit-card-fraud-detection
```

2. Install dependencies:
```
pip install pandas numpy tensorflow scikit-learn imbalanced-learn matplotlib seaborn
```

3. Run the notebook:
```
jupyter notebook Credit Card Fraud Detection.ipynb
```

### References

Dataset provided by [Machine Learning Group - ULB](https://mlg.ulb.ac.be/).
