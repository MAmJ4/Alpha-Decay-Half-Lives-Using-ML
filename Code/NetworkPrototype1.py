import numpy as np
import getData as gd
'''
This will be a 3 layer structure, 3 input neurons, 20 neurons in 1 hidden layer
and then 1 output neuron
'''
structure = [3,20,1]

# sigmoid ##################################################################################################
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# define matrix for first sets of weights to second layer using a 2d array #################################

'''w1noob = []
for x in range (structure[1]):
	w1noob.append([0,0,0])
w1 = np.array(w1noob)'''

# or:

w1 = np.asarray(np.random.rand(structure[1],structure[0]))
'''
print("Initial Weights --------------------------")
print(w1)
print(w1.shape)
'''

# define biases ############################################################################################
b1 = np.asarray(np.random.rand(structure[1],1))
'''
print("Initial Biases --------------------------")
print(b1)
print(b1.shape)
'''

# store given activations for first layer ###################################################################

Z = 92
N = 126
Q = 8.775

# min max normalisation (https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)
# (https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)

Qmin = min(gd.getQArray())
Qmax = max(gd.getQArray())

Z = (Z-80)/(118-80)
N = (N-92)/(177-92)
Q = (Q-Qmin)/(Qmax-Qmin)

'''
Z = int(input("Protons: "))
N = int(input("Neutrons: "))
Q = float(input("Energy Release: "))
'''

#print("Initial Layer 1 Activations ----------------------")
a0 = np.array([[Z],[N],[Q]])
#print(a0)

#print(a0.shape)


# compute activations of second layer #################################
a1 = np.array([])

mult = np.array([np.matmul(w1,a0)])
mult = np.reshape(mult, (20,1))

presigmoid = mult + b1

print(presigmoid)
print ("------------------")

a1 = sigmoid(presigmoid)
print(a1)
print(a1.shape)

# define matrix for second set of weights to 3rd layer using 2d array (which will be 1d anyway)
print("---------------------------------------------------")
w2 = np.asarray(np.random.rand(structure[2],structure[1]))
print (w2)
result = np.matmul(w2,a1)
print(result.shape)
print("Final Half-Life: ", result)

def feedforward (Z,N,Q,structure):
	# define matrix for 1st set of weights and initialise with random values
	w1 = np.asarray(np.random.rand(structure[1],structure[0]))
	b1 = np.asarray(np.random.rand(structure[1],1)) # define first set of biases randomly

	# min-max scale input data to get it into range [0,1]
	Z = (Z-80)/(118-80)
	N = (N-92)/(177-92)
	Q = (Q-Qmin)/(Qmax-Qmin)
	a0 = np.array([[Z],[N],[Q]]) # use scaled inputs as initial activations

	# compute second layer activations
	a1 = np.array([]) # declaring array
	mult = np.array([np.matmul(w1,a0)]) # multiply weights with activations
	mult = np.reshape(mult, (20,1)) # reshape to make 20x1 matrix
	a1 = sigmoid(mult+b1) # add biases and plug into sigmoid

	# define and randomly intialise initial weights for hidden -> final layer
	w2 = np.asarray(np.random.rand(structure[2],structure[1]))
	result = np.matmul(w2,a1) # calculate result and do not plug into activation function
	print("Final Half-Life: ", result)

feedforward(92, 126, 8.775, structure)

