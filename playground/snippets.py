import torch

def setup_gpu():
  # Use GPU if available
  device = torch.device("cuda") if torch.cuda.is_available()\
    else torch.device("cpu")

  if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Enforce all operations to be deterministic on GPU for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

  return device

if __name__=="__main__":
  # Seed random
  torch.manual_seed(1827)

  # Use GPU if available
  device = setup_gpu()
  
  # GPU settings
  setup_gpu()

  # Backpropagation
  x = torch.arange(3, dtype=torch.float32, requires_grad=True)
  # Push to device (used for GPU device)
  x = x.to(device)
  y = ((x + 2) ** 2 + 3).mean()

  y.backward()
  print(x.grad)
