import numpy as np
import data
from data import Data

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))


class Network (object):
	def __init__ (self, structure, learningrate = 0.001):
		d = Data ()
		self.size = len(structure) # set size equal to number of layers
		self.structure = structure # set structure (n of neurons) to match input
		
		self.weights = [] # declare weight array
		self.biases = [] # declare bias array
		for x in range (0, (self.size - 1)):
			# initialise weights and biases with random values
			self.weights.append(np.asarray(np.random.rand(structure[x+1],structure[x])))
			self.biases.append(np.asarray(np.random.rand(structure[x+1],1)))
		
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
		for x in range (0, self.size - 2):
			a = self.activations [x]
			mult = np.array([np.matmul(self.weights[x], a)])
			mult = np.reshape(mult, (self.structure[x+1],1))
			self.activations [x+1] = sigmoid(mult + self.biases[x])
		''' 
		manually apply last layer weights and biases to avoid sigmoid, could have
		been implemented in for loop with if statement but would have been quite messy
		'''
		af = np.array([])
		mult = np.array([np.matmul(self.weights[self.size - 2], self.activations [self.size - 2])])
		mult = np.reshape (mult, (self.structure[self.size - 1],1))
		af = mult + self.biases[self.size - 2]

		return af

	def backpropagation (self)



d = Data()
net = Network ([6,25,30,30,25,1])
print(net.feedforward ((d.getIsotope()[123])))