# EECE 6840 Homework 3
# Created on 9/16/2025
# Created by Mina Gaffney


# Shallow vs. Deep Comparison:
# Train a shallow and deep network on Fashion-MNIST. Experiment with different depths and widths.
# Plot loss, accuracy, and gradient norms across layers.
# Comment on evidence of vanishing/exploding gradients.

# importing things
import tensorflow as tf
import matplotlib.pyplot as plt

# loading fashion_mnist dataset from keras.io
# Images are 28x28 greyscale
# labels are 0-9 (10 indices)
f_mnist = tf.keras.datasets.fashion_mnist
(train_data, train_labels), (test_data, test_labels) = f_mnist.load_data()
print('Dataset Loaded')


## Building model based on prior model used in EECE 5890 class (
# From https://github.com/mgaffney-mcw/EECE_5890_AI_and_ML/blob/main/EECE_5890_TermProject.py

# Modfiying to build a shallow network with one hidden layer with 10 neurons and
# a 10 neuron output layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# Test the loss and accuracy of the model on the test data
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()



predictions = probability_model.predict(all_conf_images_test)

predictions[0]

np.argmax(predictions[0])

all_test_labels[0]

# Activation Function Experiments:
# Repeat training with ReLU, LeakyReLU, ELU, and GELU.
# Plot training/validation curves.
# Discuss which activation gave the most stable training and why.


# Batch Normalization and Residuals:
# Add BatchNorm to your deep network and repeat training.
# Add simple residual skip connections.
# Compare convergence speed and final accuracy with/without these modifications.


# Initialization and Optimizers:
# Experiment with Xavier vs. He initialization.
# Train with SGD, Adam, and AdamW.
# Report quantitative differences in convergence and test accuracy.

