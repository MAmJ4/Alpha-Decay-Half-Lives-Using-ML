import numpy as np
import data 
from data import Data

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))

sigmoid_prime = lambda z : sigmoid(z)*(1-sigmoid(z))

structure = [6,16,16,16,16,1]

d = Data ()

global weights 
weights = []
global biases 
biases = []
global activations
activations = []
global preactive 
preactive = []
global size 
size = len(structure)

for x in range (0, (size - 1)):
	weights.append(np.asarray(np.random.rand(structure[x+1],structure[x])))
	biases.append(np.asarray(np.random.rand(structure[x+1],1)))

for x in range (0, size):
	activations.append(np.zeros((structure[x],1)))

def feedforward(data, structure):
	global activations
	# min-max scaling
	Z = (data[0]-min(d.getZ()))/(max(d.getZ())-min(d.getZ()))
	N = (data[1]-min(d.getN()))/(max(d.getN())-min(d.getN()))
	A = (data[2]-min(d.getA()))/(max(d.getA())-min(d.getA()))
	Q = (data[3]-min(d.getQ()))/(max(d.getQ())-min(d.getQ()))
	Zd = (data[4]-min(d.getZDist()))/(max(d.getZDist())-min(d.getZDist()))
	Nd = (data[5]-min(d.getNDist()))/(max(d.getNDist())-min(d.getNDist()))

	activations [0] = np.array([[Z], [N], [A], [Q], [Zd], [Nd]]) # use scaled inputs as initial activations

	for x in range (0, size - 2):
		a = activations [x]
		mult = np.array([np.matmul(weights [x], a)])
		mult = np.reshape(mult, (structure [x+1],1))
		preactive.append (mult + biases [x])
		activations [x+1] = sigmoid(mult + biases[x])

	af = np.array([])
	mult = np.array([np.matmul(weights[size - 2],activations [size - 2])])
	mult = np.reshape (mult, (structure[size - 1],1))
	af = mult + biases[size - 2]
	activations.append([af])
	preactive.append ([af])

	return af

# remember, last weight = weights[size-2], last bias = biases [size-2]
# do last weight linearly, do the rest with inverse sigmoid

def backpropagation (activation, target, learningrate):

	global activations
	global biases
	global weights

	error_l = activations [-1] - target
	error_l = np.reshape (error_l, (1, structure[-1]))
	deltab = []
	deltab.insert (0, error_l)
	deltaw = []
	deltaw.insert (0, np.matmul(activations[-2], error_l))

	# second last layer
	#print (weights[-1].shape)
	#print (error_l.shape)

	foo = np.matmul (weights[-1].transpose(), error_l)
	error_2l = np.multiply (foo, sigmoid_prime (preactive[-2]))
	error_2l = np.reshape (error_2l, (1,structure[-2]))

	deltab.insert (0, error_2l)
	deltaw.insert (0, np.matmul(activations[-3], error_2l))


	# third last layer
	#print (weights[-2].transpose().shape)
	#print (error_2l.shape)
	foo = np.matmul (weights[-2].transpose(), error_2l.transpose())
	error_3l = np.multiply (foo, sigmoid_prime (preactive[-3]))
	error_3l = np.reshape (error_3l, (1, structure[-3]))
	deltab.insert (0, error_3l)
	deltaw.insert (0, np.matmul(activations[-4], error_3l))

	# fourth last layer
	foo = np.matmul (weights[-3].transpose(), error_3l.transpose())
	error_4l = np.multiply (foo, sigmoid_prime (preactive[-4]))
	error_4l = np.reshape (error_4l, (1,structure[-4]))
	deltab.insert (0, error_4l)
	deltaw.insert (0, np.matmul(activations[-5], error_4l))

	# 5th last layer
	foo = np.matmul (weights[-4].transpose(), error_4l.transpose())
	error_5l = np.multiply (foo, sigmoid_prime (preactive[-5]))
	error_4l = np.reshape (error_4l, (1,structure[-5]))
	deltab.insert (0, error_5l)
	#print (activations[-6].shape)
	#print (error_5l.shape)
	deltaw.insert (0, np.matmul (activations[-6].transpose(), error_5l))

	'''	for w,dw in zip(weights,deltaw):
		print (f"Original Shape: {w.shape}")
		print (dw.shape)
		w = w+learningrate*dw
		print (f"New Shape: {w.shape}")
		w = w.reshape ()'''

	for x in range (0,5):
		#print(f"Original Shape: {weights[x].shape}")
		weights[x] = weights[x] - learningrate * deltaw [x]
		weights[x] = np.reshape (weights[x], (structure[x+1], structure[x]))
		#print(f"New Shape: {weights[x].shape}")

	for b,db in zip (biases, deltab):
		b = b-learningrate*db
	#weights = (w + learningrate*dw for w,dw in zip (weights, deltaw))
	#biases = (b + learningrate*db for b,db in zip (biases, deltab))




i = 0
isotope = d.getIsotope()

#activation = feedforward (isotope[i], structure)
#print (activation)
#print (activations [-1])
#backpropagation (activation, np.log10 (d.getHL()[i]), 0.01)


error = feedforward (isotope[i], structure) - np.log10 (d.getHL()[i])
print (f"Error: {error}")

for z in range (0,10):
	#activation = feedforward (isotope[i], structure)
	backpropagation (feedforward (isotope[i], structure), np.log10 (d.getHL()[i]), 0.01)
	print (f"{z+1} iterations done")

error = feedforward (isotope[i], structure) - np.log10 (d.getHL()[i])
print (f"Error: {error}")