# EECE 6840 Final Project
# Created on 10/21/2025
# Created by Mina Gaffney

# importing things
#import tensorflow as tf
#import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import pandas as pd
import os
import numpy as np

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

# TODO: determine how to best organize iORG and pORG signals for each cone so that it is easy to feed into NN
# Indexing to keep track of cones is going to be a real pain... debating on how to best manage..
common_coords = pd.merge(iORG_dataset, pORG_dataset, on=['X', 'Y'], how='inner')

# TODO: Transform coords so everything is in the same pixel space
# looks like the coordinates are from their respective images and not from a common ROI despite the overlap
# Need to transform the coordinates so that everything is wrt the same pixel coordinates


# TODO: to CNN or to RNN, that is the question...?




# First loading in the data:
dataset=pd.read_excel(fName, sheet_name=['AOIP','OCVL'])
AOIP_df=dataset['AOIP']
OCVL_df=dataset['OCVL']


