import matplotlib.pyplot as plt
import matplotlib
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchvision

from const import NUM_WORKERS
from loader import Loader
from utils import download_pretrained_models, setup_gpu

def init_visualisers():
  # Matplotlib colour map
  plt.set_cmap("cividis")

  # set_matplotlib_formats("svg", "pdf")
  # Matplotlib styling
  matplotlib.rcParams["lines.linewidth"] = 2.0

  # Seaborn is a visualiser like Matplotlib
  sns.set()

# Examples of augmented images
def show_example_images():
  NUM_IMAGES = 6
  imgs = torch.stack(
    [img for idx in range(NUM_IMAGES) for img in unlabeled_data[idx][0]],
    dim=0
  )
  img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True,
    pad_value=0.9)
  img_grid = img_grid.permute(1, 2, 0)

  plt.figure(figsize=(10, 5))
  plt.title("Augmented image examples of the STL10 dataset")
  plt.imshow(img_grid)
  plt.axis("off")
  plt.show()

if __name__=="__main__":
  init_visualisers()
  pl.seed_everything(42)

  device = setup_gpu()

  print("Device:", device)
  print("Number of workers:", NUM_WORKERS)

  download_pretrained_models()

  # Load data
  # WARNING - 3GB of data - this will take a while!
  loader = Loader()
  train_data_contrast = loader.load_train()
  unlabeled_data = loader.load_unlabeled()

  # show_example_images()

  # Continue:
  # "Finally, now that we have discussed all details, let’s implement SimCLR below as a PyTorch Lightning module:""
