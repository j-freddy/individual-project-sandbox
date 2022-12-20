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
