import numpy as np
import torch

# https://python.pages.doc.ic.ac.uk/2022/lessons/pytorch/03-torch-tensors/index.html

if __name__=="__main__":
  
  # Tensor: Example 1

  tensor = torch.Tensor([
    [1, 2],
    [3, 4]
  ])
  print(tensor)

  # Tensor: Example 2

  tensor = torch.zeros((3, 3, 3)) + 2
  print(tensor)

  # Tensor: Example 3

  tensor = torch.from_numpy(np.array([
      [1, 2],
      [3, 4],
      [5, 6],
  ], dtype=float))
  print(tensor.mean(axis=0))
  print(tensor.mean(axis=1))
