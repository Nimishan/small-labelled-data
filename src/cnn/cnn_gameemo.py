from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dropout, GlobalAveragePooling1D, Dense

# NOTE: Plug in the same preprocessing block (data loading, scaling, encoding,
# 95/5 split, reshape to (samples, features, 1)) before using this model definition.

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu'),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# model = create_cnn_model((X_train.shape[1], 1), len(np.unique(y_train)))
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
