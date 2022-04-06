### Importing Libraries ###
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns


### Ignoring CUDA Devices ###
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # https://stackoverflow.com/questions/70106418/how-can-i-run-tensorflow-without-gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


### Getting Dataset and Splitting Dataset ###
column_names = ['Element', 'Z (Protons)', 'N (Neutrons)', 'Nucleons', 'Energy', 'Half-Life', 'ELDM']

dataset = pd.read_csv("Database.csv", names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0) # splitting the dataset into testing and training, 80% is training
test_dataset = dataset.drop(train_dataset.index) # using inbuilt tensorflow options to speed up the prototype

train_labels = (train_dataset.copy()).pop('Half-Life') # setting the label I want to train to find
test_labels = (test_dataset.copy()).pop('Half-Life') # have to copy dataset as I don't want to pop these from the main dataset


### Normalisation of Parameters ###
normaliser = tf.keras.layers.Normalization(axis=-1) # creating a normaliser layer
train_features = train_dataset.copy() # creating a copy of training dataset to perform calculations on

normaliser.adapt(np.array(train_features)) # the layer will compute a mean and variance for each element in the np array features

#printing the effects of normalisation:

print("Initial Values: ", np.array(train_features[:1]))
print("Normalised: ", normaliser(np.array(train_features[:1])).numpy)



### Multiple-Input Single "Layer" Regression ###


#print(train_labels)
#print(dataset)
#print ("shut up james") # Debugging