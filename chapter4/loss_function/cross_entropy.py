import numpy as np

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


# 2を正解とする
t = 2

# 2の確率が最も高い場合
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# 7の確率が最も高い場合
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))