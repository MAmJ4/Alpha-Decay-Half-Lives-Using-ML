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
ZN = gd.getAArray()
t12 = gd.gett12Array()
ELDM = gd.getELDMArray()
NdivZ = gd.getAdivBArray (N,Z)

t12range = gd.getRangeArray(t12, 9.9999, -6, 6)

print("Showing Plot...")

x = Z
y = t12range

print (len(y))
y = y [:212]

print (len(y))

print (len(x))

#plt.style.use("dark_background")
plt.scatter(x, y, s=0.5, c = "blue")
plt.yscale('log')
plt.xlabel(f"{x}")
plt.ylabel(f"{y}")
plt.title(f"Range of 1 vs Protons") 
plt.show()






'''import matplotlib.pyplot as plt

gd = Data ()

Z = gd.getZ()
N = gd.getN()
Q = gd.getQ()
ZN = gd.getA()
t12 = gd.getHL()
ELDM = gd.getModel()

x = ZN
y = Q 

plt.scatter(x, y, s=0.5, c = "blue")
#plt.yscale('log')
plt.xlabel(f"{x}")
plt.ylabel(f"{y}")
plt.title(f"{x} against {y}") # was previously done manually
plt.show()
'''