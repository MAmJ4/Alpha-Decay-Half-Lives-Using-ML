import Network
from Network import Network

def getSets ():
	global stddevModel

	isotopes = d.getIsotope()
	halflives = d.getHL()

	numTrain = len(isotopes)*8//10 # number of training isotopes = len(dataset) * 0.8
	numTest = len(isotopes)-numTrain # number of testing is what remains

	totalNums = np.arange(len(isotopes)) # all the possible indices I can use for isotopes

	randomNums = np.random.choice (len(isotopes), numTrain, replace = False) # replace: whether or not a sample is returned to the sample pool
	testingNums = np.delete(totalNums, randomNums) # define testing indices to be whatever is left after getting rid of training

	trainSet = []
	trainLabels = []

	for x in range (numTrain):
		trainSet.append (isotopes[randomNums[x]])
		trainLabels.append (np.log10(halflives[randomNums[x]]))

	testSet = []
	testLabels = []

	for x in range (numTest):
		testSet.append (isotopes[testingNums[x]])
		testLabels.append (np.log10(halflives[testingNums[x]]))

	model = d.getModel()
	errors = []

	for x in range (numTest):
		actual = np.log10(halflives[testingNums[x]])
		models = np.log10(model[testingNums[x]])
		error = models - actual
		errors.append(error**2)

	stddevModel =  (np.sum(errors)**(1/2)) / (numTest)
	#print (f"Statistical Model Error: {stddevModel}")

	return trainSet, trainLabels, testSet, testLabels, stddevModel

trainSet, trainLabels, testSet, testLabels, stddevModel = getSets ()

net = Network ([6,16,20,20,16,1])

print ("Training...")
net.train(trainSet, trainLabels, epochs = 500) # train network
print ("Training Complete")

print ("Round 1")
stddevNetwork = net.evaluate(testSet, testLabels) # evaluate network on given test set
print (f"Network Deviation: {stddevNetwork}") # print net score
print (f"Stat Model Deviation: {stddevModel}") # print model score

if stddevNetwork < stddevModel: # if net is better than model
	print("Round 2") # round 2
	trainSet, trainLabels, testSet, testLabels, stddevModel = getSets () # fresh sets
	stddevNetwork = net.evaluate(testSet, testLabels) # evaluate using already trained weights and biases
	print (f"Network Deviation: {stddevNetwork}") # print net score
	print (f"Stat Model Deviation: {stddevModel}") # print model score

	if stddevNetwork < stddevModel:
		print("Round 3")
		trainSet, trainLabels, testSet, testLabels, stddevModel = getSets () # repeat above
		stddevNetwork = net.evaluate(testSet, testLabels)
		print (f"Network Deviation: {stddevNetwork}")
		print (f"Stat Model Deviation: {stddevModel}")

		if stddevNetwork < stddevModel: # if net better than model 3 times
			print ("Saving")
			net.save() # save weights and biases







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