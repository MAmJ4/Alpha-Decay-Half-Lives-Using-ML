### Importing Libraries ### ##################################################

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns


### Ignoring CUDA Devices ### ##################################################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # https://stackoverflow.com/questions/70106418/how-can-i-run-tensorflow-without-gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

### Getting Dataset ### ########################################################

column_names = ['Element', 'Z (Protons)', 'N (Neutrons)', 'Nucleons', 'Energy', 'Half-Life', 'ELDM']

dataset = pd.read_csv("Database.csv", names=column_names,
                          na_values='?', comment='\t',
                          sep=', ', skipinitialspace=True)

#dataset.drop('Element', inplace = True, axis = 1) # axis = 1 means drop column not row
del dataset ['Element']

### Splitting Dataset ### #######################################################

train_dataset = dataset.sample(frac=0.8, random_state=0) # splitting the dataset into testing and training, 80% itraining
test_dataset = dataset.drop(train_dataset.index) # using inbuilt tensorflow options to speed up the prototypes 

train_labels = (train_dataset.copy()).pop('Half-Life') # setting the label I want to train to find
test_labels = (test_dataset.copy()).pop('Half-Life') # have to copy dataset as I don't want to pop these from the main dataset

### Normalisation of Parameters ### #############################################

normaliser = tf.keras.layers.Normalization(axis=-1) # creating a normaliser layer
train_features = train_dataset.copy() # creating a copy of training dataset to perform calculations on

normaliser.adapt(np.array(train_features).astype('float32')) # the layer will compute a mean and variance for each element in the np array features

#printing the effects of normalisation:
print("Initial Values: ", np.array(train_features[:1]))
print("Normalised: ", normaliser(np.array(train_features[:1])).numpy())

### Multiple-Input Regression ### ###############################################

# Will create a 2 layer model will the first being the normalisation layer and the second being the calculation

# Sequential Docs: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# units defines dimensionality of the output space, it will be 1 dimension
# using Sequential as it has exactly 1 input tensor and 1 output tensor

linear_model.predict(train_features[:10])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)