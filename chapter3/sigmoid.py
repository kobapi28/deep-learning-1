import numpy as np
# この2行によって savefig() が動作するように
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.savefig('sigmoid.png')
