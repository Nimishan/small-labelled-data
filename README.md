# EEG-Based Emotion Recognition with Limited Data

This repository contains the implementation of shallow and deep learning models used to
investigate data-volume-specific recommendations for EEG-based emotion recognition under
limited data availability, as presented in our paper (link/citation to be added upon publication).

## Repository Structure

```
EEG-Emotion-Recognition/
├── src/
│   ├── svm/                     # Support Vector Machine
│   ├── decision_tree/           # Decision Tree
│   ├── random_forest/           # Random Forest
│   ├── knn/                     # K-Nearest Neighbors
│   ├── logistic_regression/     # Logistic Regression
│   ├── mlp/                     # Multi-Layer Perceptron
│   ├── cnn/                     # Convolutional Neural Network
│   ├── lstm/                    # Long Short-Term Memory
│   ├── gru/                     # Gated Recurrent Unit / Conv1D variant
│   └── kan/                     # Kolmogorov-Arnold Network
├── requirements.txt
└── README.md
```

## Models

| Model | Type | Notes |
|---|---|---|
| SVM | Shallow | RBF kernel, C=100, gamma=0.001 |
| Decision Tree | Shallow | Gini criterion, max_depth=10 |
| Random Forest | Shallow | 100 estimators |
| KNN | Shallow | k=3, Manhattan distance |
| Logistic Regression | Shallow | L1 penalty, SAGA solver |
| MLP | Deep | 2 hidden layers with dropout |
| CNN | Deep | 1D convolutions |
| LSTM | Deep | 2-layer stacked LSTM |
| GRU (Conv1D variant) | Deep | Conv1D + MaxPooling |
| KAN | Deep | Per-feature univariate decomposition |

## Datasets

- **EEG Brainwave Dataset**
- **GAMEEMO**
- **LUMED**

Note: Raw datasets are not included in this repository due to size and licensing restrictions.
Please refer to the original dataset sources cited in the paper.

## Evaluation Protocol

- 95% / 5% train-test split
- 5-fold cross-validation on the training set (mean accuracy ± standard deviation reported)
- Held-out test accuracy and normalized confusion matrices reported per model

## Requirements

See `requirements.txt`. Core dependencies: `scikit-learn`, `tensorflow`, `pandas`, `numpy`,
`seaborn`, `matplotlib`.

## Citation

If you use this code, please cite our paper:

```
[Paper citation to be added upon publication]
```
