from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

# NOTE: This file mirrors the shallow-model pipeline used for SVM/DT/RF/KNN
# (data loading, scaling, encoding, 95/5 split, 5-fold CV) — plug in the same
# preprocessing block from svm_gameemo.py before this section if running standalone.

# Hyperparameter grid for Logistic Regression
param_grid = {
    'C': [10000],
    'penalty': ['l1'],   # Regularization type
    'solver': ['saga']   # Optimization algorithm
}

logreg_model = LogisticRegression(C=10000, penalty='l1', solver='saga', max_iter=5000)

# Example usage (after X_train, y_train are prepared as in svm_gameemo.py):
# cv_scores = cross_val_score(logreg_model, X_train, y_train, cv=5, scoring='accuracy')
# print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} (SD)")
# logreg_model.fit(X_train, y_train)
# accuracy = logreg_model.score(X_test, y_test)
# print(f"Held-out Test Accuracy: {accuracy:.4f}")
