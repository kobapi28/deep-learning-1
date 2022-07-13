import numpy as np


def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x) # xと同じ形状の配列を生成

  # f(x+h) - f(x-h) / 2*h をそれぞれに対してやってるだけ
  for idx in range(x.size):
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x)

    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val

  return grad

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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a-c) # オーバーフロー対策
  sum_exp_a = np.sum(exp_a)
  return exp_a / sum_exp_a



class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 重みの初期化
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  # 認識・推論を行う
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

  # x: 入力, t: 教師データ
  def loss(self, x, t):
    y = self.predict(x)
    return cross_entropy_error(y, t)


  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y==t) / float(x.shape[0])
    return accuracy

  # 重みパラメータに対する勾配をもとめる
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)
    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads