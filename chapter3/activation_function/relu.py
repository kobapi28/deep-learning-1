import numpy as np
# この2行によって savefig() が動作するように
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def relu(x):
  return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-0.1, 5)
plt.savefig('relu.png')