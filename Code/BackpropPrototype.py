import numpy as np
import data
from data import Data
'''
This will be a 3 layer structure, 3 input neurons, 20 neurons in 1 hidden layer
and then 1 output neuron
'''

structure = [3,25, 1]

gd = Data()

# sigmoid ##################################################################################################
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

Qmin = min(gd.getQ())
Qmax = max(gd.getQ())

w1 = np.asarray(np.random.rand(structure[1],structure[0])) # fix subtract 1 to allow negative weights
b1 = np.asarray(np.random.rand(structure[1],1)) # define first set of biases randomly

w2 = np.asarray(np.random.rand(structure[2],structure[1])) # fix subtract 1 to allow negative weights
b2 = np.asarray(np.random.rand(structure[2],1)) # define first set of biases randomly

result = 0

def feedforward (Z,N,Q,structure):
	# define matrix for 1st set of weights and initialise with random values
	#w1 = np.asarray(np.random.rand(structure[1],structure[0])) # fix subtract 1 to allow negative weights
	#b1 = np.asarray(np.random.rand(structure[1],1)) # define first set of biases randomly

	# min-max scale input data to get it into range [0,1]
	Z = (Z-80)/(118-80)
	N = (N-92)/(177-92)
	Q = (Q-Qmin)/(Qmax-Qmin)
	a0 = np.array([[Z],[N],[Q]]) # use scaled inputs as initial activations

	# compute second layer activations
	a1 = np.array([]) # declaring array
	mult = np.array([np.matmul(w1,a0)]) # multiply weights with activations
	mult = np.reshape(mult, (structure[1],1)) # reshape to make 20x1 matrix # fix 20 with structure[1]
	a1 = sigmoid(mult+b1) # add biases and plug into sigmoid # fix subtract 1 to allow for neg activations

	# define and randomly intialise initial weights for hidden -> final layer
	#w2 = np.asarray(np.random.rand(structure[2],structure[1]))
	result = np.matmul(w2,a1) + b2 # calculate result and do not plug into activation function
	print("Final Half-Life: ", result)

#feedforward(92, 126, 8.775, structure)

learningrate = 0.01




def backprop (target):
	outputError = target - result
	grad = learningrate * outputError
	hidden_T = np.transpose(w2)
	w2 += np.matmul(hidden_T, grad)

	hiddenError = np.matmul (outputError, np.transpose(w2))
	grad = learningrate * hiddenError * sigmoid (a1) * (1-sigmoid(a1))
	w1 += np.matmul (grad, np.transpose (a0))


w1 = np.asarray (np.random.rand(structure[1],structure[0])) # fix subtract 1 to allow negative weights
b1 = np.asarray (np.random.rand(structure[1],1)) # define first set of biases randomly

w2 = np.asarray (np.random.rand(structure[2],structure[1])) # fix subtract 1 to allow negative weights
b2 = np.asarray (np.random.rand(structure[2],1)) # define first set of biases randomly

w2_T = np.transpose(w2)


def train (Z,N,Q,structure,target):
	for x in range (1000):
		feedforward (Z,N,Q,structure)
		backprop (target)
	feedforward(Z,N,Q,structure)



train (92, 126, 8.775, structure, 5.1e-4)

'''
hiddenErrors = np.dot(self.outputs, np.transpose(self.weightsHiddenToOutput))
27.	        # calculate gradients  
28.	        gradients = (self.learningRate * hiddenErrors * sigmoid(self.hidden) * (1 -   
29.	        sigmoid(hidden))  
30.	        # adjust weights  
31.	        self.weightsInputToHidden += np.dot(gradients, np.transpose(self.inputs))  

'''	

