import numpy as np

A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])

# 次元数確認
print(A.shape)
print(B.shape)

# 行列の積
print(np.dot(A,B))

# --- 出力 ---
# (2, 3)
# (3, 2)
# [[22 28]
#  [49 64]]