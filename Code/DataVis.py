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
t12range = gd.getRangeArray(t12, 0.5, -6, 6)

print("Showing Plot...")

x = ZN
y = Q

#plt.style.use("dark_background")
plt.scatter(x, y, s=0.5, c = "blue")
#plt.yscale('log')
plt.xlabel(f"{x}")
plt.ylabel(f"{y}")
plt.title(f"{x} against {y}")
plt.show()