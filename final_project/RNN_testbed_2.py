import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define constants
NUM_TRAIN_SAMPLES = 10
NUM_TEST_SAMPLES = 5
TIMESTEPS = 100
NUM_FEATURES = 1 # Each time series is a single feature

# Generate synthetic training data
# Shape: (num_train_samples, timesteps, num_features)
X_train = np.random.rand(NUM_TRAIN_SAMPLES, TIMESTEPS, NUM_FEATURES)

# Generate synthetic training labels
# Shape: (num_train_samples, timesteps, num_features)
y_train = np.random.rand(NUM_TRAIN_SAMPLES, TIMESTEPS, NUM_FEATURES)

# Generate synthetic test data
# Shape: (num_test_samples, timesteps, num_features)
X_test = np.random.rand(NUM_TEST_SAMPLES, TIMESTEPS, NUM_FEATURES)

# Loop through each of the 10 training samples to create separate figures
for i in range(NUM_TRAIN_SAMPLES):
    # Create a new figure with two subplots side-by-side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot X_train data on the first panel
    ax1.plot(X_train[i, :, 0])
    ax1.set_title(f'X_train - Sample {i + 1}')
    ax1.set_xlabel('Timepoints')
    ax1.set_ylabel('Value')
    ax1.grid(True)

 # Plot y_train data on the second panel
    ax2.plot(y_train[i, :, 0])
    ax2.set_title(f'y_train - Sample {i+1}')
    ax2.set_xlabel('Timepoints')
    ax2.set_ylabel('Value')
    ax2.grid(True)

    # Add an overall title for the figure
    fig.suptitle(f'Data Comparison for Sample {i+1}', fontsize=16)

    # Adjust the layout so titles and labels don't overlap
    plt.tight_layout()



# Build the RNN model
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, NUM_FEATURES)),
    layers.Dense(NUM_FEATURES)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print a summary of the model architecture
model.summary()

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=20, batch_size=2)

# Loop through each of the 5 test samples and predict
print("\nPredicting signals for the 5 test samples...")
for i in range(NUM_TEST_SAMPLES):
    # Get a single test sample
    test_sample = X_test[i:i + 1]

    # Predict the signal for this test sample
    predicted_signal = model.predict(test_sample)

    # Print the shape of the predicted output and some values
    print(f"\nPrediction for test sample {i + 1}:")
    print(f"  Input shape: {test_sample.shape}")
    print(f"  Predicted signal shape: {predicted_signal.shape}")
    print(f"  First 5 predicted values: {predicted_signal[0, :5, 0]}")





# Loop through each of the 5 test samples
for i in range(NUM_TEST_SAMPLES):
    # Get a single test sample in batch format (add batch dimension)
    test_sample = X_test[i:i + 1]

    # Predict the signal for this test sample
    # Note: This requires the 'model' object to be trained from previous steps
    try:
        predicted_signal = model.predict(test_sample)
    except NameError:
        print("Model is not defined. Cannot generate predictions.")
        break  # Stop the loop if model isn't available

    # --- Plotting Code ---
    # Create a new figure with two subplots side-by-side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot X_test data on the first panel
    # We use test_sample[0] to remove the batch dimension for plotting
    ax1.plot(test_sample[0, :, 0])
    ax1.set_title(f'X_test Input - Sample {i + 1}')
    ax1.set_xlabel('Timepoints')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    # Plot predicted_signal on the second panel
    # We use predicted_signal[0] to remove the batch dimension for plotting
    ax2.plot(predicted_signal[0, :, 0])
    ax2.set_title(f'Predicted Signal - Sample {i+1}')
    ax2.set_xlabel('Timepoints')
    ax2.set_ylabel('Value')
    ax2.grid(True)

    # Add an overall title for the figure
    fig.suptitle(f'Input vs. Prediction for Test Sample {i+1}', fontsize=16)

    # Adjust the layout so titles and labels don't overlap
    plt.tight_layout()

plt.show()