import torch

def f(x: torch.Tensor):
  return x ** 2 + 4 * x

def gradient_descent(input: torch.Tensor, steps: int=20, alpha: float=0.1):
  x = input
  print(x)

  p = torch.nn.Parameter(x)
  optimiser = torch.optim.SGD(params=[p], lr=alpha)
  # optimiser = torch.optim.Adam(params=[p], lr=alpha)
  # optimiser = torch.optim.RMSprop(params=[p], lr=alpha)

  for i in range(steps):
    # Reset p
    optimiser.zero_grad()
    # Calculate gradient
    loss = f(p)
    # Backpropagate
    loss.backward()

    # Update parameters
    optimiser.step()
    print(f"Iteration {i}: {x.item()}")
  
  return x

if __name__=="__main__":
  start = 10

  x = gradient_descent(torch.Tensor([start]), steps=30, alpha=0.2)
  y = f(x)

  # (-2, -4)
  print(f"Minimum point: {x.item(), y.item()}")
