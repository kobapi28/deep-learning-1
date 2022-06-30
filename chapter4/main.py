import numpy as np
from mnist import load_mnist

# one-hot表現 (正解となるラベルだけが1で、残りが0となるデータ構造)で取得
(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)

# 60000のデータセット内から10個データを引き抜く
train_size = x_train.shape[0]
print(train_size)
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]