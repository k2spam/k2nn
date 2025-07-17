import numpy as np
from PIL import Image

# layers
class Dense:
  def __init__(self, in_features, out_features):
    self.w = np.random.randn(in_features, out_features)
    self.b = np.zeros((1, out_features))

  def __call__(self, x):
    return x @ self.w + self.b


# activation
class ReLu:
  def __call__(self, x):
    return np.maximum(0, x)

class Sigmoid:
  def __call__(self, x):
    return 1 / (1 + np.exp(-x))


# network
class Network:
  def __init__(self):
    self.layers = [
      Dense(2, 19),
      ReLu(),
      Dense(19, 35),
      Sigmoid(),
      Dense(35, 21),
      ReLu(),
      Dense(21, 3),
      Sigmoid()
    ]

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  

# generate
def generate_image(width, height, network):
  img = Image.new('RGB', (width, height))
  pixels = img.load()

  for y in range(height):
    for x in range(width):
      nx = x / width
      ny = y / height
      input = np.array([[nx, ny]])
      output = network.forward(input)[0]
      r, g, b = (output * 256).astype(np.uint8)
      pixels[x, y] = (r, g, b)
  
  img.save('neural.png')


net = Network()
generate_image(256, 256, net)