import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

class Network (object):
	def __init__ (self, structure)
		self.num_layers = len(structure)
		self.structure = structure
		...  