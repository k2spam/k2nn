import numpy as np

class Dense:
  def __init__(self, in_features, out_features):
    self.w = np.random.randn(in_features, out_features)   # weights
    self.b = np.zeros((1, out_features))                  # biases

  def forward(self, x):
    self.x = x
    return x @ self.w + self.b

  def backward(self, grad_output, lr):
    dx = grad_output @ self.w.T
    dw = self.x.T @ grad_output
    db = grad_output

    self.w -= lr * dw
    self.b -= lr * db

    return dx
  
class ReLu:
  def forward(self, x):
    self.mask = x > 0
    return self.mask * x

  def backward(self, grad_output, lr):
    return grad_output * self.mask
  
class MSE:
  def forward(self, y_pred, y_true):
    self.y_pred = y_pred
    self.y_true = y_true
    return np.mean((y_pred - y_true) ** 2)

  def backward(self):
    return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
  
class Network:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, grad, lr):
    for layer in reversed(self.layers):
      grad = layer.backward(grad, lr)


data = [(1, 5), (2, 8), (3, 11)]

net = Network([
  Dense(1, 10),
  ReLu(),
  Dense(10, 5),
  ReLu(),
  Dense(5, 1)
])

mse = MSE()
lr = 0.01

for epoch in range(500):
  total_loss = 0
  for x_scalar, y_scalar in data:
    x = np.array([[x_scalar]])
    y = np.array([[y_scalar]])

    out = net.forward(x)
    loss = mse.forward(out, y)
    total_loss += loss
    grad = mse.backward()
    net.backward(grad, lr)
  
  if epoch % 10 == 0:
    print(f'error: {total_loss:.4f}')

test = np.array([[4]])
pred = net.forward(test)
print(f'Prediction: {pred}')