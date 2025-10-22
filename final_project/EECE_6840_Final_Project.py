# EECE 6840 Final Project
# Created on 10/21/2025
# Created by Mina Gaffney

# importing things
#import tensorflow as tf
#import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import pandas as pd
import os

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








# First loading in the data:
dataset=pd.read_excel(fName, sheet_name=['AOIP','OCVL'])
AOIP_df=dataset['AOIP']
OCVL_df=dataset['OCVL']


