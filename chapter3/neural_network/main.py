import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# 入力層
X = np.array([1.0, 0.5])
# 入力層から中間層A1への重み
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# バイアス
B1 = np.array([0.1, 0.2, 0.3])


# 中間層 A1
A1 = np.dot(X, W1) + B1
# 活性化
Z1 = sigmoid(A1)


# 中間層A1から中間層A2への重み
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# バイアス
B2 = np.array([0.1, 0.2])


# 中間層 A2
A2 = np.dot(Z1, W2) + B2
# 活性化
Z2 = sigmoid(A2)


# 中間層A2から出力層への重み
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])


# 出力層 A3
A3 = np.dot(Z2, W3) + B3
# 恒等関数
Y = A3

print(Y)