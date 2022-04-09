import numpy as np
import data 
from data import Data as d

sigmoid = lambda x : 1.0/(1.0+np.exp(-x))

structure = [6,16,16,16,16,1]

Zdist = gd.getZdistArray()
Ndist = gd.getNdistArray()
ZArr = gd.getZArray()
NArr = gd.getNArray()
AArr = gd.getAArray()
QArr = gd.getQArray()

weights = []
biases = []
activations = []
size = len(structure)

for x in range (0, (size - 1)):
	weights.append(np.asarray(np.random.rand(structure[x+1],structure[x])))
	biases.append(np.asarray(np.random.rand(structure[x+1],1)))

for x in range (0, size):
	activations.append(np.zeros((structure[x],1)))

def feedforward(Z, N, A, Q, Zd, Nd, structure):
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

	return af

i = 0

t12 = feedforward (ZArr[i], NArr[i], AArr[i], QArr[i], Zdist[i], Ndist[i], structure)
print ("Half-Life: "+ str(t12[0][0]) + "s")