import csv as csv
import numpy as np

import sys
import os
data_dir = os.path.abspath (os.path.join("..","Network and Data"))
sys.path.insert(1, data_dir) # add this path to system path

class Data ():

	def __init__ (self):
		self.data = []
		with open("Network and Data/Database.csv") as database:
			csvreader = csv.reader (database)
			for row in csvreader:
				self.data.append([row[0],int(row[1]),int(row[2]),int(row[3]),float(row[4]),float(row[5]),float(row[6])])
		
		# Format: Element, Z, N, A, Q, T12, ELDM
		for isotope in self.data:

			Zdist = min([abs(isotope[1]-2), abs(isotope[1]-8), abs(isotope[1]-20), 
				abs(isotope[1]-28), abs(isotope[1]-50), abs(isotope[1]-82), abs(isotope[1]-126)])
			isotope.insert (2, Zdist)

			Ndist = min([abs(isotope[3]-2), abs(isotope[3]-8), abs(isotope[3]-20), 
				abs(isotope[3]-28), abs(isotope[3]-50), abs(isotope[3]-82), abs(isotope[3]-84), abs(isotope[3]-126)])
			isotope.insert (4, Ndist)
		
		# Format: Element, Z, ZDist, N, NDist, A, Q, T12, ELDM


	def getZ (self):
		Z = []
		for x in self.data:
			Z.append(x[1])
		return Z

	def getZDist (self):
		ZDist = []
		for x in self.data:
			ZDist.append(x[2])
		return ZDist

	def getN (self):
		N = []
		for x in self.data:
			N.append (x[3])
		return N

	def getNDist (self):
		NDist = []
		for x in self.data:
			NDist.append(x[4])
		return NDist

	def getA (self):
		A = []
		for x in self.data:
			A.append (x[5])
		return A

	def getQ (self):
		Q = []
		for x in self.data:
			Q.append (x[6])
		return Q

	def getHL (self):
		HL = []
		for x in self.data:
			HL.append (x[7])
		return HL

	def getModel (self):
		Model = []
		for x in self.data:
			Model.append (x[8])
		return Model

	def getIsotope (self):
		Isotope = []
		for x in self.data:
			Isotope.append ([x[1], x[3], x[5], x[6], x[2], x[4]])
		return Isotope