import numpy as np
import matplotlib.pyplot as plt
import data
from data import Data
import joblib

# miscellaneous functions
sigmoid = lambda x : 1.0/(1.0+np.exp(-x))
sigmoid_prime = lambda z : sigmoid(x)*(1-sigmoid(x))

# database
d = Data()

# network class
class Network ():

	def __init__ (self, structure, learningrate = 0.01):
		d = Data ()
		self.size = len(structure) # set size equal to number of layers
		self.structure = structure # set structure (n of neurons) to match input
		
		self.weights = [] # declare weight array
		self.biases = [] # declare bias array
		for x in range (0, (self.size - 1)):
			# initialise weights and biases with random values
			self.weights.append(np.asarray(np.random.uniform(-1,1, (structure[x+1],structure[x]))))
			self.biases.append(np.asarray(np.random.uniform(-1,1, (structure[x+1],1))))
		
		self.preactive = []

		self.activations = [] # declare activations
		for x in range (0, self.size):
			# initialise activations with zeros
			self.activations.append(np.zeros((structure[x],1)))

		self.learningrate = learningrate

	def feedforward (self, data):

		# min-max normalise all values
		Z = (data[0]-min(d.getZ()))/(max(d.getZ())-min(d.getZ()))
		N = (data[1]-min(d.getN()))/(max(d.getN())-min(d.getN()))
		A = (data[2]-min(d.getA()))/(max(d.getA())-min(d.getA()))
		Q = (data[3]-min(d.getQ()))/(max(d.getQ())-min(d.getQ()))
		Zd = (data[4]-min(d.getZDist()))/(max(d.getZDist())-min(d.getZDist()))
		Nd = (data[5]-min(d.getNDist()))/(max(d.getNDist())-min(d.getNDist()))
	
		# use normalised inputs as initial activations
		self.activations [0] = np.array([[Z], [N], [A], [Q], [Zd], [Nd]]) 

		# iterate through hidden layers using weights and biases (except for last)
		for x in range (0, self.size - 1):
			active = self.activations [x]
			mult = np.array([np.matmul(self.weights[x], active)])
			mult = np.reshape(mult, (self.structure[x+1],1))
			presigmoid = mult + self.biases [x]
			self.preactive.append(presigmoid)
			self.activations [x+1] = sigmoid(mult + self.biases[x])

		#manually apply last layer weights and biases to avoid sigmoid
		actfinal = np.array([])
		mult = np.array([np.matmul(self.weights[self.size - 2], self.activations [self.size - 2])])
		mult = np.reshape (mult, (self.structure[self.size - 1],1))
		actfinal = mult + self.biases[self.size - 2]

		self.activations[-1] = [actfinal] # was self.activations.append([af]) 
		self.preactive.append ([actfinal])

		return actfinal

	def backpropagation (self, activation, target):

		deltaBiases = [] # define arrays for change in biases
		deltaWeights = [] # define arrays for change in weights

		learningrate = self.learningrate

		# error in last layer
		error = activation - target # getting error
		error = np.reshape (error, (self.structure[-1], 1)) # reshaping error to be array
		deltaBiases.insert (0, error) # error = delta biases
		deltaWeights.insert (0, np.matmul(error, self.activations[self.size-2].transpose())) # weight delta is error propagated backwards # WAS HARDCODED

		for x in range (0, self.size - 2):
			prop = np.matmul (self.weights[-(x+1)].transpose(), error) # propagate error backwards
			error = np.multiply (prop, sigmoid_prime (self.preactive[-(x+3)])) # hadamard product with preactivationss
			error = np.reshape (error, (self.structure[-(x+2)] , 1)) # reshape because numpy is peculiar
			deltaBiases.insert (0, error) # error = delta bias so add to start
			deltaWeights.insert (0, np.matmul(error, self.activations[((self.size-3)-x)].transpose())) # add delta weights

		for x in range (0,self.size-1):
			self.weights [x] = self.weights [x] - (learningrate * deltaWeights[x]) # adjust each weight by delta weight
			self.biases[x] = self.biases [x] - (learningrate * deltaBiases[x]) # adjust each bias by delta bias

	def train (self, dataset, targets, epochs):
		for x in range (epochs): # for the specified amount of epochs
			#print (f"Epoch {x+1}")
			for y in range (0, len(dataset)): # for each value in the dataset
				activation = self.feedforward (dataset[y]) # feed it forward
				self.backpropagation (activation, targets[y]) # backpropagate against the target

	def evaluate (self, dataset, targets):
		predictions = [] # define array for predictions
		errors = [] # define array for errors
		for isotope in dataset:
			prediction = self.feedforward (isotope)
			predictions.append(prediction) # add predictions to array of predictions
		for x in range (len(predictions)):
			error = predictions[x] - targets[x] # calculate errors
			errors.append (error**2) # append error^2 to array
		stddev =  (np.sum(errors)**(1/2)) / len(dataset) # sigma = (1/n)*(sum of (errors)^2)^(1/2)
		return float(stddev)

	def save (self):
		joblib.dump(self.weights, "weights")
		joblib.dump(self.biases, "biases")