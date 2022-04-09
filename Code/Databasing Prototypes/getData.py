import csv
data = []

with open("Database.txt") as dataB:
	csvreader = csv.reader(dataB)
	lineCount = 0
	for row in csvreader:
		if lineCount == 0:
			lineCount+=1
		else:	
			data.append([row[0],int(row[1]),int(row[2]),int(row[3]),float(row[4]),float(row[5]),float(row[6])])

# Format: Element, Z, N, Z+N, Q, T12, ELDM

def getZArray ():
	Z = []
	for x in data:
		Z.append(x[1])
	return Z

def getNArray():
	N = []
	for x in data:
		N.append(x[2])
	return N

def getAArray():
	A = []
	for x in data:
		A.append(x[3])
	return A

def getQArray():
	Q = []
	for x in data:
		Q.append(x[4])
	return Q

def gett12Array():
	t12 = []
	for x in data:
		t12.append(x[5])
	return t12

def getELDMArray():
	ELDM = []
	for x in data:
		ELDM.append(x[6])
	return ELDM

def getZdistArray():
	Zdist = []
	for x in data:
		dist = min([abs(x[1]-2), abs(x[1]-8), abs(x[1]-20), abs(x[1]-28), abs(x[1]-50), abs(x[1]-82), abs(x[1]-126)])
		Zdist.append(dist)
	return Zdist

def getNdistArray():
	Ndist = []
	for x in data:
		dist = min([abs(x[2]-2), abs(x[2]-8), abs(x[2]-20), abs(x[2]-28), abs(x[2]-50), abs(x[2]-82), abs(x[2]-84), abs(x[2]-126)])
		Ndist.append(dist)
	return Ndist

def getRangeArray (t12, rangeSize, expLow, expEnd):
	t12range = [] # Initialise Array
	y=expLow # Lowest Exponent
	z=1 # Iterator from 1 to 9 for mantissa
	for x in t12:
		while (y<expEnd):
			while (z<9.99999): # to bypass floating point errors
				stepstart = float(f"{z}e{y}") # lower bound is z x10^ y 
				stepend = float(f"{z+rangeSize}e{y}") # upper bound is z+rangeSize x10^ y
				#print(f"From {stepstart} to {stepend}")
				if (stepstart <= x < stepend): # If that particular arraay value is within this range
					t12range.append(stepstart) # make its value the lower bound
					break # stop the loop
				z+=rangeSize
			y+=1 # iterate exponent
			z=1
		y=expLow
		z=1
	return t12range

def getAdivBArray (A,B):
	if len(A) != len(B):
		print ("Arrays must be of same size!")
		return
	else:
		AdivB = []
		for x in range (0, len(A)):
			AdivB.append((A[x])/(B[x]))
		return AdivB
