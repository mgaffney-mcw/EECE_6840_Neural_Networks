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
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')])
#
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
#
# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
#
# # Test the loss and accuracy of the model on the test data
# test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)
#
# # 2. Use tf.GradientTape and 3. Calculate gradients
# gradient_norms = []
# layer_names = [layer.name for layer in model.layers if layer.trainable_weights]
#
# with tf.GradientTape() as tape:
#     predictions = model(train_data)
#     loss = tf.keras.losses.SparseCategoricalCrossentropy()(train_labels, predictions)
#
# # Get gradients for all trainable weights
# trainable_weights = [weight for layer in model.layers for weight in layer.trainable_weights]
# grads = tape.gradient(loss, trainable_weights)
#
# # 4. Calculate gradient norms for each layer
# grad_idx = 0
# for layer in model.layers:
#     if layer.trainable_weights:
#         layer_grads = []
#         for weight in layer.trainable_weights:
#             layer_grads.append(grads[grad_idx])
#             grad_idx += 1
#         # Compute L2 norm for the layer's gradients
#         # Flatten and concatenate all gradients for a layer before computing the norm
#         flat_grads = tf.concat([tf.reshape(g, [-1]) for g in layer_grads if g is not None], axis=0)
#         if tf.size(flat_grads) > 0:
#             norm = tf.norm(flat_grads)
#             gradient_norms.append(norm.numpy())
#         else:
#             gradient_norms.append(0.0) # No gradients for this layer (e.g., if no trainable weights)
#
# # 5. Plot the norms
# plt.figure(figsize=(10, 6))
# plt.bar(layer_names, gradient_norms)
# plt.xlabel("Layer")
# plt.ylabel("Gradient Norm (L2)")
# plt.title("Gradient Norms Across Layers")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.draw()
#
# plt.figure(figsize=(10, 6))
# plt.plot(gradient_norms)
# plt.title('Gradient Norm')
# plt.ylabel('Gradient Norm')
# plt.xlabel('Layer')
# plt.draw()
#
# # Plot training & validation accuracy values
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
#
# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# plt.show()


# Modfiying to build a deep network with 3 hidden layers with 10 neurons each and
# a 10 neuron output layer
model_deep = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])


model_deep.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

probability_model_deep = tf.keras.Sequential([model_deep,
                                         tf.keras.layers.Softmax()])

# Test the loss and accuracy of the model on the test data
test_loss, test_acc = model_deep.evaluate(test_data,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# 2. Use tf.GradientTape and 3. Calculate gradients
gradient_norms = []
layer_names = [layer.name for layer in model_deep.layers if layer.trainable_weights]

with tf.GradientTape() as tape:
    predictions = model_deep(train_data)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(train_labels, predictions)

# Get gradients for all trainable weights
trainable_weights = [weight for layer in model_deep.layers for weight in layer.trainable_weights]
grads = tape.gradient(loss, trainable_weights)

# 4. Calculate gradient norms for each layer
grad_idx = 0
for layer in model_deep.layers:
    if layer.trainable_weights:
        layer_grads = []
        for weight in layer.trainable_weights:
            layer_grads.append(grads[grad_idx])
            grad_idx += 1
        # Compute L2 norm for the layer's gradients
        # Flatten and concatenate all gradients for a layer before computing the norm
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in layer_grads if g is not None], axis=0)
        if tf.size(flat_grads) > 0:
            norm = tf.norm(flat_grads)
            gradient_norms.append(norm.numpy())
        else:
            gradient_norms.append(0.0) # No gradients for this layer (e.g., if no trainable weights)

# 5. Plot the norms
plt.figure(figsize=(10, 6))
plt.bar(layer_names, gradient_norms)
plt.xlabel("Layer")
plt.ylabel("Gradient Norm (L2)")
plt.title("Gradient Norms Across Layers")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.draw()

plt.figure(figsize=(10, 6))
plt.plot(gradient_norms)
plt.title('Gradient Norm')
plt.ylabel('Gradient Norm')
plt.xlabel('Layer')
plt.draw()

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

# Comment on evidence of vanishing/exploding gradients.

# For my single layer neuron the hidden layer looks like there was some
# evidence of vanishing gradients due to how small the gradient norms for
# the hidden layer was however, that was recovered with the softmax output
# layer. (Of note my hidden layer used ReLU so that may be why the gradients
# were being pushed into the dead region) For the deep network I did not see
# evidence of vanishing gradients. The gradients grew across layers until the
# very last layer where they came down a little.




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

