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


def gradient_descent(f, init_x, lr, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad

  return x



# sample
# f(x0, x1) = x0^2 + x1^2
def function1(x):
  return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function1, init_x, lr=0.1, step_num=100))

# 学習率が大きすぎる例
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function1, init_x, lr=10, step_num=100))

# 学習率が小さすぎる例
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function1, init_x, lr=1e-10, step_num=100))

