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
import tkinter as tk
from tkinter import ttk

# modified from geeks for geeks example
# testing out a drop down menu
root = tk.Tk()
root.geometry("200x200")

# Global variable to store the combobox value
selected_value = None

def on_submit():
    """Gets the value from the combobox and closes the window."""
    global selected_value
    selected_value = cb.get()
    print(f"Selected value saved: {selected_value}")
    root.destroy()


# Dropdown options
a = ["Shallow NN", "Deep NN", "ReLU", "Leaky ReLu", "ELU", "GELU", "Leaky Batch Norm"]

# Combobox
cb = ttk.Combobox(root, values=a)
cb.set("Select an NN")
cb.pack()


# Button to display selection
tk.Button(root, text="Save and Quit", command=on_submit).pack()

root.mainloop()


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
if selected_value == 'Shallow NN':
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

    # 2. Use tf.GradientTape and 3. Calculate gradients
    gradient_norms = []
    layer_names = [layer.name for layer in model.layers if layer.trainable_weights]

    with tf.GradientTape() as tape:
        predictions = model(train_data)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(train_labels, predictions)

    # Get gradients for all trainable weights
    trainable_weights = [weight for layer in model.layers for weight in layer.trainable_weights]
    grads = tape.gradient(loss, trainable_weights)

    # 4. Calculate gradient norms for each layer
    grad_idx = 0
    for layer in model.layers:
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


# # Modfiying to build a deep network with 3 hidden layers with 10 neurons each and
# # a 10 neuron output layer
if selected_value == 'Deep NN' or selected_value == 'ReLU':
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

###################################################################################

# Activation Function Experiments:
# Repeat training with ReLU, LeakyReLU, ELU, and GELU.
# Plot training/validation curves.
# Discuss which activation gave the most stable training and why.

# original implementation was ReLU... moving on to the other activation functions
# starting with ELU because it's easy to swap out with my existing dense layers
if selected_value == 'ELU':
    model_deep = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(10, activation='softmax')])


    model_deep.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

    # probability_model_deep = tf.keras.Sequential([model_deep,
    #                                          tf.keras.layers.Softmax()])

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


# on to GELU which is also easy to implement...
if selected_value == 'GELU':
    model_deep = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='gelu'),
        tf.keras.layers.Dense(10, activation='gelu'),
        tf.keras.layers.Dense(10, activation='gelu'),
        tf.keras.layers.Dense(10, activation='softmax')])


    model_deep.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

    # probability_model_deep = tf.keras.Sequential([model_deep,
    #                                          tf.keras.layers.Softmax()])

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

# # Leaky relu
if selected_value == 'Leaky ReLu':
    model_deep = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05)),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05)),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05)),
        tf.keras.layers.Dense(10, activation='softmax')])


    model_deep.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

    # probability_model_deep = tf.keras.Sequential([model_deep,
    #                                          tf.keras.layers.Softmax()])

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

# Discuss which activation gave the most stable training and why.

# Leaky ReLU gave the most stable training because it reached the highest
# accuracy (0.827) and the accuracy plot shows it reached an accuracy of
# around 0.8 after the first epoch, after that the accuracy continued to
# climb near exponentially until around epoch 4 where it started to reach
# are more linear increase in training accuracy.

###########################################################################

# Batch Normalization and Residuals:
# Add BatchNorm to your deep network and repeat training.
# Add simple residual skip connections.
# Compare convergence speed and final accuracy with/without these modifications.

# Adding batch normalization improved the accuracy of the leaky ReLU model.
# The original test accuracy (without batch norm) was 0.827 and with batch
# normalization the accuracy was improved to 0.85. In terms of convergence
# speed, the batch normalization model reached a higher accuracy after epoch
# 1 and the overall curve appeared to have a steeper slope than the model without
# batch normalization. (Although they both performed well and the improvement
# is marginal.)

# Leaky relu with batch normalization in first layer
if selected_value == 'Leaky Batch Norm':
    model_deep = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05)),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05)),
        tf.keras.layers.Dense(10, activation='softmax')])


    model_deep.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

    # probability_model_deep = tf.keras.Sequential([model_deep,
    #                                          tf.keras.layers.Softmax()])

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

###########################################################################

# Initialization and Optimizers:
# Experiment with Xavier vs. He initialization.
# Train with SGD, Adam, and AdamW.
# Report quantitative differences in convergence and test accuracy.

# Report quantitative differences in convergence and test accuracy.

# The combination of optimizer and initialization with the highest
# accuracy was Xavier initialization paired with the Adam optimizer
# (accuracy = 0.842). That being said, the Adam and AdamW optimizers
# paired with the He and Xavier initializations were all very similar
# (and had accuracies around 0.83-0.84). Although, He and AdamW was a
# better combination compared to He and Adam. The stochastic gradient
# descent optimizer failed across the board with test accuracies around
# 0.1 regardless of the initialization used. Both cases experienced
# vanishing gradients.


# # # Leaky relu with Xavier intialization.
# # # According to the googlewebs glorot_uniform = Xavier
# model_deep = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05),kernel_initializer='glorot_uniform'),
#     tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05),kernel_initializer='glorot_uniform'),
#     tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05),kernel_initializer='glorot_uniform'),
#     tf.keras.layers.Dense(10, activation='softmax')])
#
#
# model_deep.compile(optimizer='adamw',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# # Train the model
# history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
#
# # probability_model_deep = tf.keras.Sequential([model_deep,
# #                                          tf.keras.layers.Softmax()])
#
# # Test the loss and accuracy of the model on the test data
# test_loss, test_acc = model_deep.evaluate(test_data,  test_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)
#
# # 2. Use tf.GradientTape and 3. Calculate gradients
# gradient_norms = []
# layer_names = [layer.name for layer in model_deep.layers if layer.trainable_weights]
#
# with tf.GradientTape() as tape:
#     predictions = model_deep(train_data)
#     loss = tf.keras.losses.SparseCategoricalCrossentropy()(train_labels, predictions)
#
# # Get gradients for all trainable weights
# trainable_weights = [weight for layer in model_deep.layers for weight in layer.trainable_weights]
# grads = tape.gradient(loss, trainable_weights)
#
# # 4. Calculate gradient norms for each layer
# grad_idx = 0
# for layer in model_deep.layers:
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


# Leaky relu with He intialization.
model_deep = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05),kernel_initializer='he_normal'),
    tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05),kernel_initializer='he_normal'),
    tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.05),kernel_initializer='he_normal'),
    tf.keras.layers.Dense(10, activation='softmax')])


model_deep.compile(optimizer='adamw',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model_deep.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# probability_model_deep = tf.keras.Sequential([model_deep,
#                                          tf.keras.layers.Softmax()])

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