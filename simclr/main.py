import matplotlib.pyplot as plt
import matplotlib
import pytorch_lightning as pl
import seaborn as sns

from const import NUM_WORKERS
from utils import download_pretrained_models, setup_gpu

def init_visualisers():
  # Matplotlib colour map
  plt.set_cmap("cividis")

  # set_matplotlib_formats("svg", "pdf")
  # Matplotlib styling
  matplotlib.rcParams["lines.linewidth"] = 2.0

  # Seaborn is a visualiser like Matplotlib
  sns.set()

if __name__=="__main__":
  init_visualisers()
  pl.seed_everything(42)

  device = setup_gpu()

  print("Device:", device)
  print("Number of workers:", NUM_WORKERS)

  download_pretrained_models()
