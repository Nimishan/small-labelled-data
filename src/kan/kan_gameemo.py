from tensorflow.keras.layers import Input, Dense, add
from tensorflow.keras.models import Model

# NOTE: Plug in the same preprocessing block (data loading, scaling, encoding,
# 95/5 split) from svm_gameemo.py before using this model definition.

def create_kan_model(input_shape):
    inputs = Input(shape=(input_shape,))

    # Decompose into univariate functions
    univariate_layers = [Dense(64, activation='relu')(inputs[:, i:i + 1]) for i in range(input_shape)]

    # Aggregate transformation across all channels
    aggregated_layers = [Dense(64, activation='relu')(layer) for layer in univariate_layers]
    summed_layers = add(aggregated_layers)

    # Final output layer
    output = Dense(3, activation='softmax')(summed_layers)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# model = create_kan_model(X_train.shape[1])
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
