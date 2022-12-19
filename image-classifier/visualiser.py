from matplotlib import pyplot as plt
import numpy as np

def imshow(img):
  # Unnormalise
  img = img / 2 + 0.5
  plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
  plt.show()
