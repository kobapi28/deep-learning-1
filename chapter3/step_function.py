import numpy as np
# この2行によって savefig() が動作するように
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def step_function(x):
  return np.array(x>0, dtype=np.int64)

# -5から5までの範囲で0.1刻みに配列生成
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.savefig('step_function.png')