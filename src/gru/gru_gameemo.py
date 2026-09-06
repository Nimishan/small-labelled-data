from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# NOTE: Plug in the same preprocessing block (data loading, scaling, encoding,
# 95/5 split, reshape to (samples, features, 1)) before using this model definition.

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# model = create_model((X_train.shape[1], 1), len(np.unique(y_train)))
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
