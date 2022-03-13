import getData as gd
import os
import sys
import subprocess

try:
	import matplotlib.pyplot as plt
except ImportError:
	devnull = open(os.devnull,"w")
	print("Installing MatPlotLib")
	subprocess.run([sys.executable,"-m", "pip", "install", 
					"matplotlib"], stdout = devnull, stderr=devnull)

Z = gd.getZArray()
N = gd.getNArray()
Q = gd.getQArray()
ZN = gd.getZNArray()
t12 = gd.gett12Array()
ELDM = gd.getELDMArray()
NdivZ = gd.getAdivBArray (N,Z)

'''
for x in range (0, len(Z)):
	NdivZ.append((N[x])/(Z[x]))
'''
t12range = gd.getRangeArray(t12, 0.5, -6, 6)

A = [1,2]
B = [1,2,3]
AdivB = gd.getAdivBArray(A,B)







'''
print("Showing Plot...")

x = NdivZ

plt.style.use("dark_background")
plt.scatter(x, t12range, s=0.5, c = "white")
plt.yscale('log')
plt.xlabel(f"{x}")
plt.ylabel("Half-Life Range")
plt.title("Neutron-Proton Ratio against Half-Life Range")
plt.show()
'''