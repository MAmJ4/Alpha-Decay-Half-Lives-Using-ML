import numpy as np
import getData as gd

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))

structure = [6,10,25,30,25,10,1]

w01 = np.asarray(np.random.rand(structure[1],structure[0])) # define weights from layer 0 to 1
b01 = np.asarray(np.random.rand(structure[1],1)) # define biases from layer 0 to 1

w12 = np.asarray(np.random.rand(structure[2],structure[1]))
b12 = np.asarray(np.random.rand(structure[2],1))

w23 = np.asarray(np.random.rand(structure[3],structure[2]))
b23 = np.asarray(np.random.rand(structure[3],1))

w34 = np.asarray(np.random.rand(structure[4],structure[3]))
b34 = np.asarray(np.random.rand(structure[4],1))

w45 = np.asarray(np.random.rand(structure[5],structure[4]))
b45 = np.asarray(np.random.rand(structure[5],1))

Zdist = gd.getZdistArray()
Ndist = gd.getNdistArray()
ZArr = gd.getZArray()
NArr = gd.getNArray()
AArr = gd.getAArray()
QArr = gd.getQArray()

weights = []
biases = []
activations = []

for x in range (0,(len(structure) - 1)):
	weights.append(np.asarray(np.random.rand(structure[x+1],structure[x])))
	biases.append(np.asarray(np.random.rand(structure[x+1],1)))

#weights[0], weights[1], weights[2], weights[3], weights[4] = w01, w12, w23, w34, w45
#biases[0], biases[1], biases[2], biases[3], biases[4]= b01,b12,b23,b34,b45

for x in range (0, len(structure)):
	activations.append(np.zeros((structure[x],1)))

size = len(structure)

#print (len(weights))
#print (len(biases))
#print (len(activations))

def iterfeed(Z, N, A, Q, Zd, Nd, structure):
	# min-max scaling
	Z = (Z-min(ZArr))/(max(ZArr)-min(ZArr))
	N = (N-min(NArr))/(max(NArr)-min(NArr))
	A = (A-min(AArr))/(max(AArr)-min(AArr))
	Q = (Q-min(QArr))/(max(QArr)-min(QArr))
	Zd = (Zd-min(Zdist))/(max(Zdist)-min(Zdist))
	Nd = (Nd-min(Ndist))/(max(Ndist)-min(Ndist))

	activations [0] = np.array([[Z], [N], [A], [Q], [Zd], [Nd]]) # use scaled inputs as initial activations

	for x in range (0, size - 2):
		a = activations [x]
		mult = np.array([np.matmul(weights[x], a)])
		mult = np.reshape(mult, (structure[x+1],1))
		activations [x+1] = sigmoid(mult + biases[x])

	af = np.array([])
	mult = np.array([np.matmul(weights[size - 2],activations [size - 2])])
	mult = np.reshape (mult, (structure[size - 1],1))
	af = mult + biases[size - 2]
	activations.append ([af])
	return af

def feedforward (Z, N, A, Q, Zd, Nd, structure):
	# min-max scaling
	Z = (Z-min(ZArr))/(max(ZArr)-min(ZArr))
	N = (N-min(NArr))/(max(NArr)-min(NArr))
	A = (A-min(AArr))/(max(AArr)-min(AArr))
	Q = (Q-min(QArr))/(max(QArr)-min(QArr))
	Zd = (Zd-min(Zdist))/(max(Zdist)-min(Zdist))
	Nd = (Nd-min(Ndist))/(max(Ndist)-min(Ndist))

	a0 = np.array([[Z], [N], [A], [Q], [Zd], [Nd]]) # use scaled inputs as initial activations

	a1 = np.array([]) # declaring array
	mult = np.array([np.matmul(w01,a0)]) # multiply weights with activations
	mult = np.reshape(mult, (structure[1],1)) # reshape to make 20x1 matrix 
	a1 = sigmoid(mult+b01) # add biases and plug into sigmoid # fix subtract 1 to allow for neg activations

	a2 = np.array([])
	mult = np.array([np.matmul(w12,a1)])
	mult = np.reshape (mult, (structure[2],1))
	a2 = sigmoid (mult + b12)

	a3 = np.array([])
	mult = np.array([np.matmul(w23,a2)])
	mult = np.reshape (mult, (structure[3],1))
	a3 = sigmoid (mult + b23)

	a4 = np.array([])
	mult = np.array([np.matmul(w34,a3)])
	mult = np.reshape (mult, (structure[4],1))
	a4 = sigmoid (mult + b34)

	a5 = np.array([])
	mult = np.array([np.matmul(w45,a4)])
	mult = np.reshape (mult, (structure[5],1))
	a5 = mult + b45

	return a5


i = 0
'''t12 = feedforward (ZArr[i], NArr[i], AArr[i], QArr[i], Zdist[i], Ndist[i], structure)
print ("Half-Life V: "+ str(t12[0][0]) + "s")

i=0'''

t12 = iterfeed (ZArr[i], NArr[i], AArr[i], QArr[i], Zdist[i], Ndist[i], structure)
print ("Half-Life I: "+ str(t12[0][0]) + "s")

