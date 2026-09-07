# EEG Emotion Recognition — Training Data Scheme Evaluation

This repository contains the experimental code accompanying a research paper evaluating the effect of training data volume on EEG-based emotion/cognitive state recognition classifiers, under two distinct train-test partitioning schemes applied to two publicly available EEG datasets.

---

## Repository Structure

Each of the four folders contains **10 individual notebooks** — one per model — so every classifier can be run, shared, or cited independently.

```
eeg-training-scheme-evaluation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── GAMEEMO_Scheme1_Variable_Test/          ← 10 notebooks
│   ├── SVM_GAMEEMO_Scheme1.ipynb
│   ├── DT_GAMEEMO_Scheme1.ipynb
│   ├── RF_GAMEEMO_Scheme1.ipynb
│   ├── LR_GAMEEMO_Scheme1.ipynb
│   ├── KNN_GAMEEMO_Scheme1.ipynb
│   ├── MLP_GAMEEMO_Scheme1.ipynb
│   ├── CNN_GAMEEMO_Scheme1.ipynb
│   ├── LSTM_GAMEEMO_Scheme1.ipynb
│   ├── GRU_GAMEEMO_Scheme1.ipynb
│   └── KAN_GAMEEMO_Scheme1.ipynb
│
├── GAMEEMO_Scheme2_Static_Test/            ← 10 notebooks
│   ├── SVM_GAMEEMO_Scheme2.ipynb
│   ├── DT_GAMEEMO_Scheme2.ipynb
│   ├── RF_GAMEEMO_Scheme2.ipynb
│   ├── LR_GAMEEMO_Scheme2.ipynb
│   ├── KNN_GAMEEMO_Scheme2.ipynb
│   ├── MLP_GAMEEMO_Scheme2.ipynb
│   ├── CNN_GAMEEMO_Scheme2.ipynb
│   ├── LSTM_GAMEEMO_Scheme2.ipynb
│   ├── GRU_GAMEEMO_Scheme2.ipynb
│   └── KAN_GAMEEMO_Scheme2.ipynb
│
├── LUMED_Scheme1_Variable_Test/            ← 10 notebooks
│   ├── SVM_LUMED_Scheme1.ipynb
│   ├── DT_LUMED_Scheme1.ipynb
│   ├── RF_LUMED_Scheme1.ipynb
│   ├── LR_LUMED_Scheme1.ipynb
│   ├── KNN_LUMED_Scheme1.ipynb
│   ├── MLP_LUMED_Scheme1.ipynb
│   ├── CNN_LUMED_Scheme1.ipynb
│   ├── LSTM_LUMED_Scheme1.ipynb
│   ├── GRU_LUMED_Scheme1.ipynb
│   └── KAN_LUMED_Scheme1.ipynb
│
└── LUMED_Scheme2_Static_Test/              ← 10 notebooks
    ├── SVM_LUMED_Scheme2.ipynb
    ├── DT_LUMED_Scheme2.ipynb
    ├── RF_LUMED_Scheme2.ipynb
    ├── LR_LUMED_Scheme2.ipynb
    ├── KNN_LUMED_Scheme2.ipynb
    ├── MLP_LUMED_Scheme2.ipynb
    ├── CNN_LUMED_Scheme2.ipynb
    ├── LSTM_LUMED_Scheme2.ipynb
    ├── GRU_LUMED_Scheme2.ipynb
    └── KAN_LUMED_Scheme2.ipynb
```

---

## Experimental Schemes

### Scheme 1 — Variable Test Data Scheme

The entire dataset is loaded into a single pool. At each step, `train_test_split` is applied with a given `train_size` (0.95 → 0.05). The complement becomes the test set automatically — **both train and test sizes change at each step**.

**Key implication:** Accuracy differences across steps reflect the combined effect of less training data *and* a differently-sized test set.

**Workflow per step:**
1. Load and concatenate all subject files into one pool.
2. Apply `train_test_split(X, y, train_size=p, random_state=42, [stratify=y])`.
3. Run 5-fold cross-validation on the training split.
4. Fit the model on the full training split.
5. Evaluate once on the resulting variable test split.

---

### Scheme 2 — Static Test Data Scheme

The dataset is divided once, upfront, into a **training pool** and a **fixed held-out test set**. The test set is loaded once and never modified. At each step only the training pool is subsampled to the chosen fraction.

**Key implication:** Since the test set never changes, any accuracy difference across steps is due *only* to the change in training data volume.

**Workflow per step:**
1. Load and concatenate subject files into a training pool.
2. Load the static test set (once, reused at every step).
3. Subsample the training pool: `data_train = train_pool.sample(frac=p)`.
4. Fit the `StandardScaler` and `LabelEncoder` on training data only; apply `transform` to the test set (no leakage).
5. Run 5-fold CV on the training split; fit the model; evaluate on the unchanged static test set.

---

## Models

| Category | Model | Abbreviation |
|----------|-------|--------------|
| Shallow ML | Support Vector Machine | SVM |
| Shallow ML | Decision Tree | DT |
| Shallow ML | Random Forest | RF |
| Shallow ML | Logistic Regression | LR |
| Shallow ML | K-Nearest Neighbours | KNN |
| Deep Learning | Multi-Layer Perceptron | MLP |
| Deep Learning | Convolutional Neural Network (1D) | CNN |
| Deep Learning | Long Short-Term Memory | LSTM |
| Deep Learning | Gated Recurrent Unit | GRU |
| Deep Learning | Kolmogorov-Arnold Network | KAN |

### Hyperparameters

| Model | Dataset | Key Parameters |
|-------|---------|---------------|
| SVM | GAMEEMO | `C=100`, `gamma=0.001`, `kernel='rbf'` |
| SVM | LUMED | `C=100`, `gamma=0.0001`, `kernel='rbf'` |
| DT | Both | `criterion='gini'`, `max_depth=10` |
| RF | Both | `n_estimators=100`, `max_depth=None` |
| LR | Both | `C=10000`, `penalty='l1'`, `solver='saga'` |
| KNN | Both | `n_neighbors=3`, `metric='manhattan'` |
| MLP | Both | Dense(64→32→10→output), Dropout(0.3), Adam, 30 epochs |
| CNN | Both | Conv1D(32,64) + GAP + Dense(128), Dropout(0.2), Adam, 30 epochs |
| LSTM | Both | LSTM(64→32) + Dense(output), Dropout(0.2), Adam, 30 epochs |
| GRU | Both | GRU(64→32) + Dense(output), Dropout(0.2), Adam, 30 epochs |
| KAN | Both | Per-feature Dense(64) branches + add + Dense(output), Adam, 30 epochs |

---

## Datasets

### GAMEEMO
- **Source:** [GAMEEMO EEG Dataset](https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo)
- **Subjects:** 28 participants, 4 games each
- **Labels:** Valence and Arousal (emotion dimensions)
- **Files:** `S{id:02}_G{game}_Denoised.csv` (e.g., `S01_G1_Denoised.csv`)
- **Drive path:** `EEG Datasets/GAMEEMO/Denoised EEG Data/`
- **Static test file (Scheme 2):** `gameemo_test.csv` in the same folder

### LUMED
- **Source:** [LUMED EEG Dataset](https://figshare.com/articles/dataset/Loughborough_University_Multimodal_Emotion_Dataset_-_2/12644033)
- **Subjects:** 13 participants
- **Labels:** Integer class labels in `label` column (stored as `"[1]"` strings)
- **Files:** `wavelet_denoised_s{id:02}.csv` (e.g., `wavelet_denoised_s01.csv`)
- **Drive path:** `EEG Datasets/LUMED/LUMED CSV/`
- **Static test file (Scheme 2):** `lumed_testset.csv` in the same folder

> **Note:** Raw datasets are not included in this repository. Download from the links above and place them in your Google Drive under the paths above, or update `folder_path` in each notebook.

---
### EEG Brainwave
- **Source:** [EEG Brainwave Dataset (Feeling Emotions)](https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/emotions.csv)
- **Format:** Single CSV file, loaded directly from a public URL — no Google Drive setup required
- **Labels:** String emotion labels in a `label` column (e.g., `"POSITIVE"`, `"NEGATIVE"`, `"NEUTRAL"`), encoded with `LabelEncoder`
- **Features:** All columns except `label`

> **Note:** The GAMEEMO and LUMED raw datasets are not included in this repository. Download them from the links above and place them in your Google Drive. The EEG Brainwave dataset is loaded directly from its public URL in the code.

## How to Run

1. Open any notebook in **Google Colab** (upload it or open from GitHub via `File → Open notebook → GitHub`).
2. Mount your Google Drive (each notebook calls `drive.mount('/content/drive')` automatically).
3. Place the dataset files under the expected Google Drive path (see above), or edit `folder_path` at the top of the notebook.
4. For **Scheme 2** notebooks, ensure the static test file is present.
5. **Run all cells.** The notebook is pre-configured for `train_size = 0.95`. Change this value and re-run to sweep across the full range (0.95 → 0.05).

## Adapting the Notebooks for the EEG Brainwave Dataset
Adapting any notebook to use the EEG Brainwave dataset requires only **3 small changes** at the top of each notebook — the model code, CV loop, and evaluation section remain identical.

### Change 1 — Replace the data loading block

Remove the Google Drive mounting and file-reading loop. Replace with:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/emotions.csv')

X = df.drop(columns=['label']).values
y_raw = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_enc = LabelEncoder()
y = label_enc.fit_transform(y_raw)
```

### Change 2 — Scheme 1 (Variable Test): update the split line

```python
# Replace the GAMEEMO/LUMED split with:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=train_size, random_state=42, stratify=y)
```

### Change 3 — Scheme 2 (Static Test): update the pool and test-set lines

Since there are no separate subject files, split the single dataframe once upfront to create a fixed test set:

```python
from sklearn.model_selection import train_test_split

# Create a fixed test set once (e.g. 20%) — reuse this across all steps
X_pool, X_test, y_pool, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)

# At each step, subsample the training pool
import numpy as np
idx = np.random.RandomState(42).choice(len(X_pool), size=int(train_size * len(X_pool)), replace=False)
X_train, y_train = X_pool[idx], y_pool[idx]
```

Everything below these lines (model definition, 5-fold CV loop, evaluation, confusion matrix) stays exactly the same.

---

## Evaluation Metrics

- **5-Fold CV Accuracy** — mean ± standard deviation on the training split at each `train_size` step.
- **Held-out Test Accuracy** — single evaluation on the test set at each step.
- **Confusion Matrix** — normalised, plotted as a heatmap (%).
- **Classification Report** (LUMED only) — precision, recall, F1-score per class.

---

## Dependencies

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.1
matplotlib>=3.5
seaborn>=0.12
tensorflow>=2.12
```

Install in Colab with:
```python
!pip install scikit-learn tensorflow pandas matplotlib seaborn
```

---


---

## License

This code is shared for academic and research purposes. Please refer to the individual dataset licences for data usage terms.
