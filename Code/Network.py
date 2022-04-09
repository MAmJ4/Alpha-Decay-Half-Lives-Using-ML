import numpy as np

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))

class Data ():

class Network (object):
	def __init__ (self, structure)
		self.size = len(structure)
		self.structure = structure
		self.weights = []
		self.biases = []
		for x in range (0, (size - 1)):
			self.weights.append(np.asarray(np.random.rand(structure[x+1],structure[x])))
			self.biases.append(np.asarray(np.random.rand(structure[x+1],1)))
		self.activations =[]
		for x in range (0, size):
			self.activations.append(np.zeros((structure[x],1)))