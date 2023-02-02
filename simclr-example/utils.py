import os
import torch
import torch.utils.data as data
from urllib.error import HTTPError
import urllib.request

from const import CHECKPOINT_PATH

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

def download_pretrained_models():
  # Pretrained models
  base_url =\
    "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial17/"

  pretrained_files = [
    "SimCLR.ckpt",
    "ResNet.ckpt",
    "tensorboards/SimCLR/events.out.tfevents.SimCLR",
    "tensorboards/classification/ResNet/events.out.tfevents.ResNet",
  ]

  pretrained_files += [
    f"LogisticRegression_{size}.ckpt"
    for size in [10, 20, 50, 100, 200, 500]
  ]

  for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)

    # Create subdirectories
    if "/" in file_path:
      os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)

    if os.path.isfile(file_path):
      print(f"{file_name} already exists")
    else:
      file_url = base_url + file_name
      print(f"Downloading {file_url}...")

      try:
        urllib.request.urlretrieve(file_url, file_path)
      except HTTPError as e:
        print(e)

def get_smaller_dataset(original_dataset, num_imgs_per_label):
  new_dataset = data.TensorDataset(
    *(t.unflatten(0, (10, 500))[:, :num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors)
  )
  return new_dataset
