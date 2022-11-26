def f(x: float):
  return x ** 2 + 4 * x

def df(x: float):
  return 2 * x + 4

def gradient_descent(input: float, steps: int=20, alpha: float=0.1):
  x = input
  print(x)

  for i in range(steps):
    x -= alpha * df(x)
    print(x)
  
  return x

if __name__=="__main__":
  # gradient_descent(10)
  x = gradient_descent(10, steps=30, alpha=0.2)
  y = f(x)

  # (-2, -4)
  print(f"Minimum point: {x, y}")
