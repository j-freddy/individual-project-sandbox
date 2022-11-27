import torch

def f(x: torch.Tensor):
  return x ** 2 + 4 * x

def gradient_descent(input: torch.Tensor, steps: int=20, alpha: float=0.1):
  x = input
  print(x)

  for i in range(steps):
    p = torch.nn.Parameter(x)
    loss = f(p)
    loss.backward()

    x -= alpha * p.grad
    print(f"Iteration {i}: {x.item()}")
  
  return x

if __name__=="__main__":
  start = 10

  x = gradient_descent(torch.Tensor([start]), steps=30, alpha=0.2)
  y = f(x)

  # (-2, -4)
  print(f"Minimum point: {x.item(), y.item()}")
