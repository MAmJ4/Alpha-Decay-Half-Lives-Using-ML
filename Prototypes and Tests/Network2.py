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

		self.trainSet, self.trainLabels, self.testSet, self.testLabels, self.modelLabels = self.getSets ()

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

		#manually apply last layer weights and biases to avoid sigmoid

		af = np.array([])
		mult = np.array([np.matmul(self.weights[self.size - 2], self.activations [self.size - 2])])
		mult = np.reshape (mult, (self.structure[self.size - 1],1))
		af = mult + self.biases[self.size - 2]

		self.activations.append ([af])
		self.preactive.append ([af])

		return af

	def backpropagation (self, activation, target):

		deltaBiases = [] # define arrays for change in biases
		deltaWeights = [] # define arrays for change in weights

		learningrate = self.learningrate

		# error in last layer
		error_l = activation - target # getting error
		error_l = np.reshape (error_l, (self.structure[-1], 1)) # reshaping error to be array
		deltaBiases.insert (0, error_l) # error = delta biases
		deltaWeights.insert (0, np.matmul(error_l, self.activations[self.size-2].transpose())) # weight delta is error propagated backwards # WAS HARDCODED

		for x in range (0, self.size - 2):
			prop = np.matmul (self.weights[-(x+1)].transpose(), error_l) # propagate error backwards
			error_l = np.multiply (prop, sigmoid_prime (self.preactive[-(x+3)])) # hadamard product with preactivationss
			error_l = np.reshape (error_l, (self.structure[-(x+2)] , 1)) # reshape because numpy is peculiar
			deltaBiases.insert (0, error_l) # error = delta bias so add to start
			deltaWeights.insert (0, np.matmul(error_l, self.activations[((self.size-3)-x)].transpose())) # add delta weights

		for x in range (0,self.size-1):
			self.weights [x] = self.weights [x] - (learningrate * deltaWeights[x]) # adjust each weight by delta weight
			self.biases[x] = self.biases [x] - (learningrate * deltaBiases[x]) # adjust each bias by delta bias
	
	def getSets (self):

		isotopes = d.getIsotope() # get set of Isotopes
		halflives = d.getHL() # get set of Half-Lives
		model = d.getModel() # get set of (Stat) Model Predictions

		numTrain = len(isotopes)*8//10 # set # of training items to be 80%
		numTest = len(isotopes) - numTrain # # of testing items is therefore 20%

		allIsotopes = np.arange(len(isotopes)) # array of numbers from 0 to len(isotopes)

		trainingNums = np.random.choice (len(isotopes), size = numTrain, replace = False) # replace: whether or not a sample is returned to the sample pool
		testingNums = np.delete(allIsotopes, trainingNums) # testingNums is all isotopes - trainingNums

		trainSet = []
		trainLabels = []
		for x in range (numTrain):
			trainSet.append (isotopes[trainingNums[x]])
			trainLabels.append (np.log10(halflives[trainingNums[x]]))

		testSet = []
		testLabels = []
		for x in range (numTest):
			testSet.append (isotopes[testingNums[x]])
			testLabels.append (np.log10(halflives[testingNums[x]]))

		modelLabels = []
		for x in range (numTest):
			modelLabels.append(np.log10(model[testingNums[x]]))

		return trainSet, trainLabels, testSet, testLabels, modelLabels

	def train (self, epochs):
		for x in range (epochs): # for the specified amount of epochs
			print (f"Epoch {x+1}")
			for y in range (0, len(self.trainSet)): # for each value in the dataset
				activation = self.feedforward (self.trainSet[y]) # feed it forward
				self.backpropagation (activation, self.trainLabels[y]) # backpropagate against the target

	def evaluate (self):
		predictions = [] # define array for predictions
		errors = [] # define array for errors
		for isotope in self.testSet:
			prediction = self.feedforward (isotope)
			predictions.append(prediction) # add predictions to array of predictions
		for x in range (len(predictions)):
			error = predictions[x] - self.testLabels[x] # calculate errors
			errors.append (error**2) # append error^2 to array
		stddev =  (np.sum(errors)**(1/2)) / len(self.testLabels) # sigma = (1/n)*(sum of (errors)^2)^(1/2)
		return float(stddev)

	def compare (self):
		errors = []
		for x in range (len(self.modelLabels)):
			error = self.testSet [x] - self.modelLabels[x]
			errors.append (error**2)

		modelScore = (np.sum(errors)**(1/2)) / len(self.modelLabels)

		netScore = self.evaluate()
		print (f"Model Score: {modelScore}")
		print (f"Net Score: {netScore}")

d = Data()
net = Network ([6,16,16,16,16,1])


print ("Training")
net.train(100) # train for 100 epochs
print ("Testing")
net.compare() # compare the model to the network