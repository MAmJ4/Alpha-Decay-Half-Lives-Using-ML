import numpy as np
import joblib

import os
import sys

data_dir = os.path.abspath (os.path.join("..","Network and Data"))
# find directory to current file
# go to parent directory ("..")
# from parent, add "Network and Data" to the path
sys.path.insert(1, data_dir) # add this path to system path

from data import Data
d = Data()

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))

weights = joblib.load("weights")
biases = joblib.load ("biases")
activations = []

structure = [6,16,20,20,16,1]
size = len(structure)
for x in range (size):
    activations.append(np.zeros((structure[x],1)))

def feedforward (data):
    # min-max normalise all values
    Z = (data[0]-min(d.getZ()))/(max(d.getZ())-min(d.getZ()))
    N = (data[1]-min(d.getN()))/(max(d.getN())-min(d.getN()))
    A = (data[2]-min(d.getA()))/(max(d.getA())-min(d.getA()))
    Q = (data[3]-min(d.getQ()))/(max(d.getQ())-min(d.getQ()))
    Zd = (data[4]-min(d.getZDist()))/(max(d.getZDist())-min(d.getZDist()))
    Nd = (data[5]-min(d.getNDist()))/(max(d.getNDist())-min(d.getNDist()))

    activations [0] = np.array([[Z], [N], [A], [Q], [Zd], [Nd]]) 


    # iterate through hidden layers using weights and biases (except for last)
    for x in range (0, size - 1):
        a = activations [x]
        mult = np.array([np.matmul(weights[x], a)])
        mult = np.reshape(mult, (structure[x+1],1))
        presig = mult + biases [x]
        activations [x+1] = sigmoid(mult + biases[x])
    
    #manually apply last layer weights and biases to avoid sigmoid
    af = np.array([])
    mult = np.array([np.matmul(weights[size - 2], activations [size - 2])])
    mult = np.reshape (mult, (structure[size - 1],1))
    af = mult + biases[size - 2]
    activations.append ([af])
    return af