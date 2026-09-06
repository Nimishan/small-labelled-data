from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# NOTE: Plug in the same preprocessing block (data loading, scaling, encoding,
# 95/5 split) from svm_gameemo.py before using this model definition.

def create_mlp_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# model = create_mlp_model(X_train.shape[1], len(np.unique(y_train)))
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
