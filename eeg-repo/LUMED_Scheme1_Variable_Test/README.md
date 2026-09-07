# LUMED — Scheme 1 — Variable Test Data Scheme

Both the training and test set sizes change at every step. `train_test_split` (with `stratify=y`) is re-applied at each `train_size` value (0.95 → 0.05).

---

## Notebooks in this folder

| File | Model | Category | Key Parameters |
|------|-------|----------|----------------|
| `SVM_LUMED_Scheme1.ipynb` | Support Vector Machine | Shallow ML | `C`, `gamma`, `kernel='rbf'` |
| `DT_LUMED_Scheme1.ipynb` | Decision Tree | Shallow ML | `criterion='gini'`, `max_depth=10` |
| `RF_LUMED_Scheme1.ipynb` | Random Forest | Shallow ML | `n_estimators=100`, `max_depth=None` |
| `LR_LUMED_Scheme1.ipynb` | Logistic Regression | Shallow ML | `C=10000`, `penalty='l1'`, `solver='saga'` |
| `KNN_LUMED_Scheme1.ipynb` | K-Nearest Neighbours | Shallow ML | `n_neighbors=3`, `metric='manhattan'` |
| `MLP_LUMED_Scheme1.ipynb` | Multi-Layer Perceptron | Deep Learning | Dense(64→32→10→out), Dropout(0.3), Adam, 10 epochs |
| `CNN_LUMED_Scheme1.ipynb` | Convolutional Neural Network | Deep Learning | Conv1D(32,64)+GAP+Dense(128), Dropout(0.2), Adam, 10 epochs |
| `LSTM_LUMED_Scheme1.ipynb` | Long Short-Term Memory | Deep Learning | LSTM(64→32)+Dense, Dropout(0.2), Adam, 10 epochs |
| `GRU_LUMED_Scheme1.ipynb` | Gated Recurrent Unit | Deep Learning | GRU(64→32)+Dense, Dropout(0.2), Adam, 10 epochs |
| `KAN_LUMED_Scheme1.ipynb` | Kolmogorov-Arnold Network | Deep Learning | Per-feature Dense(64) branches + add + Dense, Adam, 10 epochs |

---

## Dataset

**LUMED** — Subject files: `wavelet_denoised_s01.csv` … `wavelet_denoised_s13.csv`. Set `folder_path` to your Google Drive location.

**Label parsing:** Labels are stored as `"[1]"` strings. The notebooks parse them with `int(x.strip("[]"))` or `eval(x)[0]`.

---

## How to use

1. Open the desired notebook in **Google Colab**.
2. Mount your Google Drive — `drive.mount('/content/drive')` is called automatically.
3. Ensure the dataset files are at the expected path (or update `folder_path`).
4. Set `train_size` at the top of the notebook (e.g., `0.95`, `0.90`, …, `0.05`) and run all cells.
5. 5-fold CV accuracy and held-out test accuracy are printed; a confusion matrix is plotted.
