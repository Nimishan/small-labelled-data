import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

train_size = 0.95

folder_path = '/content/drive/My Drive/EEG Datasets/GAMEEMO/Denoised EEG Data/'
subject_files = [[f"S{i:02}_G{j}_Denoised.csv" for j in range(1, 5)] for i in range(1, 28)]

data_list = []
for subject_file_list in subject_files:
    for file in subject_file_list:
        file_path = os.path.join(folder_path, file)
        temp_data = pd.read_csv(file_path)
        data_list.append(temp_data)

data = pd.concat(data_list, ignore_index=True)
data = data.sample(frac=0.10, random_state=42).reset_index(drop=True)

X = data.drop(columns=['Valence', 'Arousal']).values
valence = data['Valence'].values
arousal = data['Arousal'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

valence_enc = LabelEncoder()
y_valence = valence_enc.fit_transform(valence)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_valence, test_size=1 - train_size, random_state=42
)

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print(f"5-Fold CV Accuracy with {int(train_size * 100)}% training data: {cv_mean:.4f} +/- {cv_std:.4f} (SD)")

rf_model.fit(X_train, y_train)

accuracy = rf_model.score(X_test, y_test)
print(f"Held-out Test Accuracy with {int(train_size * 100)}% training data: {accuracy:.4f}")

y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(5, 4))
sns.heatmap(cm * 100, annot=True, fmt=".2f", cmap="Blues")
plt.title(f"Confusion Matrix ({int(train_size * 100)}% Training Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
