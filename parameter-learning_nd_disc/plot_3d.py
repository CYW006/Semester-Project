import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(0,10,1)
Y = np.arange(0,10,1)
X, Y = np.meshgrid(X,Y)
Z = np.atleast_2d(np.arange(0,10,0.1)).reshape((10,10))
print(X)
print(Z)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap = 'rainbow')
plt.show()