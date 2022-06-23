import numpy as np

# numpyの配列生成
x = np.array([1.0, 2.0])
y = np.array([2.0, 4.0])

# 要素ごとに計算
# 要素数が同じでないとエラーになる
print(x + y)



# 二次元配列
# 配列の場合と同様に、同じ形状の行列どうしであれば要素ごとに計算される。
A = np.array([[1,2],[3,4]])

# 行列の掛け算もできるよ
B = np.array([10, 20])
print(A*B)