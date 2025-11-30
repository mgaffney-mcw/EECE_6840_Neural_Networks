# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np


# 1. Prepare Data
# Generate some synthetic time series data
def generate_sequence(length, n_features):
    return np.random.rand(length, n_features)


sequence_length = 50
n_features = 1
n_samples = 1000

# Create input and target sequences
# For simplicity, we'll make the target an "echo" of the input with a delay
X = np.array([generate_sequence(sequence_length, n_features) for _ in range(n_samples)])
y = np.array([np.roll(seq, -10, axis=0) for seq in X])  # Shifted sequence as target

# Split into training and validation sets
train_size = int(0.8 * n_samples)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 2. Build the Encoder-Decoder Model
# Encoder
encoder_inputs = keras.Input(shape=(sequence_length, n_features))
encoder_lstm = layers.LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(sequence_length, n_features))  # Same shape as encoder inputs for echo
decoder_lstm = layers.LSTM(64, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = layers.TimeDistributed(layers.Dense(n_features))  # Output n_features for each timestep
decoder_outputs = decoder_dense(decoder_outputs)

# Combine encoder and decoder into a full model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 3. Compile and Train the Model
model.compile(optimizer='adam', loss='mse')

# For training, we pass the input sequence to both encoder_inputs and decoder_inputs
# The decoder will learn to predict the target sequence based on the encoded context
# and its own shifted input (teacher forcing)
history = model.fit([X_train, X_train], y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=([X_val, X_val], y_val))

# 4. Make Predictions (Inference)
# For inference, the decoder needs to generate sequences step-by-step
# using its own predictions as input for the next step.
# This requires a separate inference model or a custom prediction loop.

# Encoder inference model
encoder_model = keras.Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = keras.Input(shape=(64,))
decoder_state_input_c = keras.Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# Prediction function
def predict_sequence(input_seq, n_steps_to_predict):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, sequence_length, n_features))  # Initialize with zeros
    # For a true "echo", the first input to the decoder could be the last known value
    # or a specific start-of-sequence token. Here, we'll just use the first element of the input.
    target_seq[0, 0, 0] = input_seq[0, -1, 0]  # Example: last value of input sequence

    decoded_sequence = np.zeros((1, n_steps_to_predict, n_features))

    for i in range(n_steps_to_predict):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value
        )
        decoded_sequence[0, i, 0] = output_tokens[0, 0, 0]  # Take the first predicted timestep
        states_value = [h, c]

        # Update the target sequence for the next step (teacher forcing for next prediction)
        # In a real-world scenario, you might feed the *predicted* output back as input.
        # For simplicity in this echo example, we'll just shift the output and take the first element.
        target_seq = np.roll(target_seq, -1, axis=1)
        target_seq[0, -1, 0] = output_tokens[0, 0, 0]  # Feed the predicted output back in

    return decoded_sequence

# Example prediction
# input_sequence_for_prediction = X_val[0:1]
# predicted_sequence = predict_sequence(input_sequence_for_prediction, sequence_length)
# print("Input Sequence:\n", input_sequence_for_prediction[0, :, 0])
# print("Predicted Sequence:\n", predicted_sequence[0, :, 0])