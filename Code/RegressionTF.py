import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#Your Code Here

column_names = ['Element', 'Z (Protons)', 'N (Neutrons)', 'Nucleons', 'Energy', 'Half-Life', 'ELDM']

dataset = pd.read_csv("Database.csv", names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0) # splitting the dataset into testing and training
test_dataset = dataset.drop(train_dataset.index) # using inbuilt tensorflow options to speed up the prototype

train_labels = (train_dataset.copy()).pop('Half-Life') # setting the label I want to train to find
test_labels = (test_dataset.copy()).pop('Half-Life') # have to copy dataset as I don't want to pop these from the main dataset

print(train_labels)
print(dataset)

print ("shut up james") # Debugging