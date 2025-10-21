# EECE 6840 Final Project
# Created on 10/21/2025
# Created by Mina Gaffney

# importing things
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from tkinter import ttk
import pandas as pd

# Loading in pORG and iORG dataset and sorting training vs test data
# iORG and pORG data was preprocessed, ORG signals were extracted, and
# compiled manually into 2 xlsxs
# The first two columns are the X,Y coords
# the first row is the time vector
# Subsequent rows (col 3: end) are ORG signals

# User selects dataset from file - Load in observations spreadsheet
root = Tk()
pORG_fName = filedialog.askopenfilename(title="Select compiled pORG data.", parent=root)
print('selected path: ' + fName)

if not fName:
    quit()

# First loading in the data:
dataset=pd.read_excel(fName, sheet_name=['AOIP','OCVL'])
AOIP_df=dataset['AOIP']
OCVL_df=dataset['OCVL']


