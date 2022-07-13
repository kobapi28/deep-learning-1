# 簡単なニューラルネットワークを例にした勾配を求める実装
import numpy as np
from gradient_descent import numerical_gradient

def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a-c) # オーバーフロー対策
  sum_exp_a = np.sum(exp_a)
  return exp_a / sum_exp_a

def cross_entropy_error(y, t):
	# yの次元が1だった場合
	if y.ndim == 1:
		# 2次元の配列に変換する
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0] # 要素数
	delta = 1e-7 # -∞を発生しないための対策
	# one-hot 表現ではなく、「2」などがラベルとして与えられる場合
	# arange: 連続した配列の自動作成
	# ここの詳しい解説はp95
	return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

class simpleNet:
  def __init__(self):
    # numpy.random.randn 
    # 平均0 標準偏差1 の正規分布(ガウス分布)の乱数を生成する関数
    self.W = np.random.randn(2,3) # ガウス分布で初期化

  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)
    return loss


net = simpleNet()
x = np.array([0.6, 0.9])
p = net.predict(x)
idx = np.argmax(p)
t = np.zeros(3, dtype=int)
t[idx] = 1

print(net.loss(x, t))