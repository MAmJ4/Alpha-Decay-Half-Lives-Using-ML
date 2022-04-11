import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import data
from data import Data

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))
sigmoid_prime = lambda z : sigmoid(z)*(1-sigmoid(z))


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
			a = self.activations [x]
			mult = np.array([np.matmul(self.weights[x], a)])
			mult = np.reshape(mult, (self.structure[x+1],1))
			presig = mult + self.biases [x]
			self.preactive.append(presig)
			self.activations [x+1] = sigmoid(mult + self.biases[x])

		''' 
		manually apply last layer weights and biases to avoid sigmoid, could have
		been implemented in for loop with if statement but would have been quite messy
		'''
		af = np.array([])
		mult = np.array([np.matmul(self.weights[self.size - 2], self.activations [self.size - 2])])
		mult = np.reshape (mult, (self.structure[self.size - 1],1))
		af = mult + self.biases[self.size - 2]

		self.activations.append ([af])
		self.preactive.append ([af])

		return af

	def backpropagation (self, activation, target):
		deltaBiases = []
		deltaWeights = []

		learningrate = self.learningrate

		# error in last layer
		error_l = activation - target # getting error
		error_l = np.reshape (error_l, (self.structure[-1], 1)) # reshaping error to be array
		deltaBiases.insert (0, error_l) # error = delta biases
		deltaWeights.insert (0, np.matmul(error_l, self.activations[4].transpose())) # weight delta is error propagated backwards

		for x in range (0, self.size - 2):
			prop = np.matmul (self.weights[-(x+1)].transpose(), error_l)
			error_l = np.multiply (prop, sigmoid_prime (self.preactive[-(x+3)]))
			error_l = np.reshape (error_l, (self.structure[-(x+2)] , 1))
			deltaBiases.insert (0, error_l)
			deltaWeights.insert (0, np.matmul(error_l, self.activations[((self.size-3)-x)].transpose()))

		for x in range (0,5):
			self.weights [x] = self.weights [x] - (learningrate * deltaWeights[x])
		for x in range (0,5):
			self.biases[x] = self.biases [x] - (learningrate * deltaBiases[x])

	def train (self, dataset, targets, epochs):
		for x in range (epochs):
			for y in range (0, len(dataset)):
				activation = self.feedforward (dataset[y])
				self.backpropagation (activation, targets[y]) 





d = Data()
net = Network ([6,16,16,16,16,1])
#print(net.feedforward ((d.getIsotope()[123])))

i = 100
isotopes = d.getIsotope()
halflives = d.getHL()

numTrain = len(isotopes)*8//10
numTest = len(isotopes)-numTrain

rng = default_rng

randomNums = np.random.choice (len(isotopes), numTrain, replace = False) # replace: whether or not a sample is returned to the sample pool

trainSet = []
trainLabels = []

for x in range (numTrain):
	trainSet.append (isotopes[randomNums[x]])
	trainLabels.append (halflives[randomNums[x]])

net.train(trainSet, trainLabels, 3000)

print ("Training Complete")








'''
errors = []

epochs = 2000

print (f"Error Before Backpropagation: {net.feedforward (isotopes[i]) - np.log10 (d.getHL()[i])}")

for x in range (0, epochs):
	activation = net.feedforward (isotopes[i]) - np.log10 (d.getHL()[i])
	net.backpropagation (activation, np.log10 (d.getHL()[i]))
	error = activation - np.log10 (d.getHL()[i])
	errors.append(float(error))
	if x == (epochs-1):
		print (f"Final Error in BackProp: {error}")
		print (f"Log 10 of Predicted Half Life = {activation}")
		print (f"Log 10 of Actual Half Life = {np.log10 (d.getHL()[i])}")

print (f"Error After Backpropagation: {float(net.feedforward (isotopes[i])) - float(np.log10 (d.getHL()[i]))}")
'''