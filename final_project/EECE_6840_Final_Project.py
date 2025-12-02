# EECE 6840 Final Project
# Created on 10/21/2025
# Created by Mina Gaffney

# importing things
from tkinter import Tk, filedialog
import pandas as pd
import os
import numpy as np
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from matplotlib import use
import matplotlib.pyplot as plt
#use("Qt5Agg")
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Loading in pORG and iORG dataset and sorting training vs test data
# iORG and pORG data was preprocessed, ORG signals were extracted, and
# compiled manually into 2 xlsxs
# The first two columns are the X,Y coords
# the first row is the time vector
# Subsequent rows (col 3: end) are ORG signals

# User selects input directory containing pORG and iORG compiled spreadsheets
root = Tk()
ORG_pName = filedialog.askdirectory(title="Select folder containing compiled ORG data.", parent=root)
print('selected path: ' + ORG_pName)

if not ORG_pName:
    quit()

# Loading in the data

pName_contents = os.listdir(ORG_pName)

for i in pName_contents:
    if 'iORG' in i:
        iORG_dataset = pd.read_excel(os.path.abspath(os.path.join(ORG_pName, i)))
    elif 'pORG' in i:
        pORG_dataset = pd.read_excel(os.path.abspath(os.path.join(ORG_pName, i)))
    else:
        raise ValueError("iORG or pORG must be specified within the filename of input data!")

# Organizing and cleaning steps prior to folding into model

# First we need to ensure the cone coords are comperable between the iORG and pORG data
# It looks like the coords from the iORG data have been rounded to int but the pORG coords have not

# Rounding the pORG coords to nearest int
for idx, val in pORG_dataset['X'].items():
    #print(idx, val)
    pORG_dataset.loc[idx, 'X'] = int(round(pORG_dataset.loc[idx, 'X']))
    pORG_dataset.loc[idx, 'Y'] = int(round(pORG_dataset.loc[idx, 'Y']))
pORG_dataset['X'] = pORG_dataset['X'].astype('Int64')
pORG_dataset['Y'] = pORG_dataset['Y'].astype('Int64')

# per 10/22 discussion with RFC -- Match iORG and pORG signals at the time of stimulus delivery (0)
# Truncate iORG so that it encompasses the same amount of time in sec as pORG. Then interpolate pORG
# signals such that the time iterations match. (ie: everything is on the exact same time scale as if
# acquired simultaneously)

# Looking for the start time, end time, and time interval for the pORG data first...
pORG_startTime = pORG_dataset.columns[3]
pORG_endTime = pORG_dataset.columns[-1]
pORG_intTime = pORG_dataset.columns[4] - pORG_dataset.columns[3]

# and now doing same for the iORG data...
iORG_startTime = iORG_dataset.columns[3]
iORG_endTime = iORG_dataset.columns[-1]
iORG_intTime = iORG_dataset.columns[4] - iORG_dataset.columns[3]

# Downsampling the iORG data to match the pORG data
newiORGX = pORG_dataset.columns[3:-1]

# first lets just extract the iORG signals independent of their cone coords
iORG_extractedData = iORG_dataset.loc[:, iORG_dataset.columns[3:-1]]
pORG_extractedData = pORG_dataset.loc[:, pORG_dataset.columns[3:-1]]

# pre-allocating df to save resampled data
resampled_iORG = pd.DataFrame(columns = newiORGX, data=np.empty(shape = (len(iORG_extractedData), newiORGX.size)))
resampled_iORG[:] = np.nan

# This is a dumb way to resample the iORG data but it does work so it's good enough for now
for index, row in iORG_extractedData.iterrows():
    currentCone = iORG_extractedData.loc[index,:]
    f = currentCone.reindex(currentCone.index.union(newiORGX))
    f2 = f.interpolate('index')
    f3 = f2.reindex(newiORGX)
    resampled_iORG.loc[index] = f3
    del currentCone, f, f2, f3


# Training/Test split: 1000 cones for training and 614 cones
training_input = resampled_iORG.sample(n=1000, random_state = 42)
label_input = pORG_extractedData.sample(n=1000, random_state = 42)

test_input = resampled_iORG.drop(training_input.index)
ground_truth_test = pORG_extractedData.drop(label_input.index)

# Plot a subset of the training and the label input
training_plot = training_input.sample(n=10, random_state = 44)
label_plot = label_input.sample(n=10, random_state = 44)

# for ii in range(10):
#     # Create a new figure with two subplots side-by-side (1 row, 2 columns)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
#     # Plot X_train data on the first panel
#     ax1.plot(training_plot.iloc[ii])
#     ax1.set_title(f'Training - Sample {ii + 1}')
#     ax1.set_xlabel('Timepoints')
#     ax1.set_ylabel('Value')
#     ax1.grid(True)
#
#  # Plot y_train data on the second panel
#     ax2.plot(label_plot.iloc[ii])
#     ax2.set_title(f'Label - Sample {ii+1}')
#     ax2.set_xlabel('Timepoints')
#     ax2.set_ylabel('Value')
#     ax2.grid(True)
#
#     # Add an overall title for the figure
#     fig.suptitle(f'Data Comparison for Sample {ii+1}', fontsize=16)
#
#     # Adjust the layout so titles and labels don't overlap
#     plt.tight_layout()
#
# plt.show()

# TODO: build RNN
# reformating things so that TF will accept the training data...
# Expected input must be either a tensor or a numpy array with
# the dimensions of (samples, timestamps, features)
# for this data that would ne # cones, length of time vector, and 1

# casting to numpy array and reshaping...
train_array = training_input.values
train_array[np.isnan(train_array)]=0
training_data = train_array.reshape((train_array.shape[0], train_array.shape[1], 1))

label_array = label_input.values
label_array[np.isnan(label_array)]=0
labels = label_array.reshape((label_array.shape[0], label_array.shape[1], 1))

test_array = test_input.values
test_array[np.isnan(test_array)]=0
test_data = test_array.reshape((test_array.shape[0], test_array.shape[1], 1))

truth_array = ground_truth_test.values
truth_array[np.isnan(truth_array)]=0
ground_truth = truth_array.reshape((truth_array.shape[0], truth_array.shape[1], 1))


# Build the RNN model
model = keras.Sequential([
    layers.GRU(64, return_sequences=True, input_shape=(training_input.shape[1], 1)),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=["accuracy"])

# Print a summary of the model architecture
model.summary()

# Train the model
print("Training the model...")
history = model.fit(training_data, labels, epochs=50, batch_size=2)

train_loss = history.history['loss']
epochs = range(1, len(train_loss) + 1)


#Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)


# A function to calculate and print the metrics
def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    mae = mean_absolute_error(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    rmse = np.sqrt(mse)
    # Calculate Pearson's correlation coefficient
    #pearson_r, _ = pearsonr(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    #print(f"Pearson's R: {pearson_r[0]:.4f}")

    return mse, mae, rmse

# Get all test predictions at once for efficiency
Y_test = model.predict(test_data)

print("Overall test set evaluation:")
mse, mae, rmse = evaluate_predictions(ground_truth, Y_test)

# Calculate Pearson's correlation coefficient

#pearson_r, _ = pearsonr(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
# for i, row in truth_array
#     pearson_r, _ = pearsonr(truth_array[i,:], )

# TODO: convert Y_test into an array for visualization

Pred_outputs = np.squeeze(Y_test)
resaved_pred = pd.DataFrame(columns = newiORGX, data=Pred_outputs)

# Plot a subset of the training and the label input
test_plot = test_input.sample(n=10, random_state = 50)
pred_plot = resaved_pred.sample(n=10, random_state = 50)
truth_plot = ground_truth_test.sample(n=10, random_state = 50)

for ii in range(10):
    # Create a new figure with two subplots side-by-side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot X_train data on the first panel
    ax1.plot(test_plot.iloc[ii])
    ax1.set_title(f'Test - Sample {ii + 1}')
    ax1.set_xlabel('Time from stimulus onset (sec)')
    ax1.set_ylabel('Value')
    ax1.grid(True)

 # Plot y_train data on the second panel
    ax2.plot(pred_plot.iloc[ii], label='Model Prediction')
    ax2.plot(truth_plot.iloc[ii], label = 'Ground Truth')
    ax2.set_title(f'Predicted - Sample {ii+1}')
    ax2.set_xlabel('Time from stimulus onset (sec)')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)

    # Add an overall title for the figure
    fig.suptitle(f'Model Comparison for Sample {ii+1}', fontsize=16)

    # Adjust the layout so titles and labels don't overlap
    plt.tight_layout()

plt.show()





