import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

# Define training size
train_size = 0.95  # 95% for training

# Load dataset efficiently
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

# Separate features and labels
X = data.drop(columns=['Valence', 'Arousal']).values
valence = data['Valence'].values
arousal = data['Arousal'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
valence_enc = LabelEncoder()
y_valence = valence_enc.fit_transform(valence)

# Set fixed SVM parameters
svm_model = SVC(C=100, gamma=0.001, kernel='rbf')

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_valence, test_size=1 - train_size, random_state=42
)

# ---- 5-Fold Cross-Validation on the training data ----
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print(f"5-Fold CV Accuracy with {int(train_size * 100)}% training data: {cv_mean:.4f} +/- {cv_std:.4f} (SD)")

# Fit the model on the full training data
svm_model.fit(X_train, y_train)

# Evaluate on the test data
accuracy = svm_model.score(X_test, y_test)
print(f"Held-out Test Accuracy with {int(train_size * 100)}% training data: {accuracy:.4f}")

# Confusion Matrix
y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(5, 4))
sns.heatmap(cm * 100, annot=True, fmt=".2f", cmap="Blues")
plt.title(f"Confusion Matrix ({int(train_size * 100)}% Training Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
