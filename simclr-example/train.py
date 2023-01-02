import os
import matplotlib.pyplot as plt
import matplotlib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import seaborn as sns
import torch
import torch.utils.data as data
import torchvision

from const import NUM_WORKERS
from loader import Loader
from const import CHECKPOINT_PATH
from simclr import SimCLR
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

def train_simclr(batch_size, max_epochs=500, **kwargs):
  # PyTorch lightning trainer
  # See example in pytorch-lightning repo
  # Purpose: Encapsulates gradient descent, backprop etc. in 1 function:
  #   trainer.fit
  # 
  # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
  # 
  # Note: pl.Trainer only works because we have a PyTorch Lightning module:
  #   SimCLR(pl.LightningModule)

  trainer = pl.Trainer(
    # Where the logging and weights is stored
    default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
    # TODO Deprecated
    gpus=1 if str(device) == "cuda:0" else 0,
    # Stops training when number of epochs reached
    max_epochs=max_epochs,
    # What to log (I think this is the log output that gets piped to
    # @default_root_dir)
    callbacks=[
      ModelCheckpoint(
        save_weights_only=True,
        mode="max",
        monitor="val_acc_top5"
      ),
      LearningRateMonitor("epoch"),
    ],
  )

  # Do not need optional logging
  trainer.logger._default_hp_metric = None 

  # Check if pretrained model exists
  pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")

  if os.path.isfile(pretrained_filename):
    print(f"Found pretrained model at {pretrained_filename}, loading...")
    # Automatically loads the model with saved hyperparameters
    model = SimCLR.load_from_checkpoint(pretrained_filename)
  else:
    train_loader = data.DataLoader(
      unlabeled_data,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
      # "If you load your samples in the Dataset on CPU and would like to push
      # it during training to the GPU, you can speed up the host to device
      # transfer by enabling pin_memory"
      # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
      pin_memory=True,
      num_workers=NUM_WORKERS,
    )

    val_loader = data.DataLoader(
      train_data_contrast,
      batch_size=batch_size,
      shuffle=False,
      drop_last=False,
      pin_memory=True,
      num_workers=NUM_WORKERS,
    )

    pl.seed_everything(42)

    model = SimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    
    # Load best checkpoint after training
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

  return model

if __name__=="__main__":
  init_visualisers()
  pl.seed_everything(42)

  device = setup_gpu()

  print("Device:", device)
  print("Number of workers:", NUM_WORKERS)

  # Load data
  # WARNING - 3GB of data - this will take a while!
  loader = Loader()
  train_data_contrast = loader.load_train()
  unlabeled_data = loader.load_unlabeled()

  # show_example_images()

  # Train model
  simclr_model = train_simclr(
    batch_size=256, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=500
  )
