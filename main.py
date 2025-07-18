import numpy as np
from PIL import Image

# layers
class Dense:
  def __init__(self, input_size, output_size):
    self.w = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    self.b = np.zeros((1, output_size))

  def forward(self, x):
    self.input = x
    self.output = x @ self.w + self.b
    return self.output
  
  def backward(self, grad_output, lr):
    grad_input = grad_output @ self.w.T
    grad_w = self.input.T @ grad_output
    grad_b = grad_output.sum(axis=0)

    self.w -= lr * grad_w
    self.b -= lr * grad_b

    return grad_input


# activation
class ReLu:
  def forward(self, x):
    self.input = x
    return np.maximum(0, x)
  
  def backward(self, grad_output, lr):
    return grad_output * (self.input > 0)

class Sigmoid:
  def forward(self, x):
    x = np.clip(x, -60, 60)
    self.out = 1 / (1 + np.exp(-x))
    return self.out
  
  def backward(self, grad_output, lr):
    return grad_output * self.out * (1 - self.out)
  

# error
class MSE:
  def forward(self, prediction, target):
    self.pred = prediction
    self.target = target
    return np.mean((prediction - target) ** 2)

  def backward(self):
    return 2 * (self.pred - self.target) * self.target.shape[0]


# network
class Network:
  def __init__(self):
    self.layers = [
      Dense(2, 32),
      ReLu(),
      Dense(32, 32),
      ReLu(),
      Dense(32, 3),
      Sigmoid()
    ]
    self.loss = MSE()

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x
  
  def backward(self, grad, lr):
    for layer in reversed(self.layers):
      grad = layer.backward(grad, lr)
    return grad
  
  def train(self, data, epochs=1000, lr=0.01):
    for epoch in range(epochs):
      np.random.shuffle(data)
      x = np.array([i[0] for i in data])
      y = np.array([i[1] for i in data])
      pred = self.forward(x)
      loss_val = self.loss.forward(pred, y)
      grad = self.loss.backward()
      self.backward(grad, lr)

      if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {loss_val:.5f}')
  
# load image

def load_image(path, size=(64, 64)):
  img = Image.open(path).convert('RGB').resize(size)
  width, height = img.size
  pixels = np.array(img) / 255
  data = []

  for y in range(height):
    for x in range(width):
      nx = x / width
      ny = y / height
      r, g, b = pixels[y, x]
      data.append(((nx, ny), (r, g, b)))
  
  return data, width, height

# generate image
def generate_image(network, width, height, filename='copy.png'):
  img = Image.new('RGB', (width, height))
  pixels = img.load()

  for y in range(height):
    for x in range(width):
      nx = x / width
      ny = y / height
      input = np.array([[nx, ny]])
      output = network.forward(input)[0]
      r, g, b = (output * 255).astype(np.uint8)
      pixels[x, y] = (r, g, b)
  
  img.save(filename)


data, width, height = load_image("train_64x64.png")
net = Network()
net.train(data, 3000, 0.005)
generate_image(net, width, height)