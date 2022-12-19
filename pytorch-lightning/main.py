import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

# Defining the encoder and decoder neural network models
# (This is not important)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

class LitAutoEncoder(pl.LightningModule):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  # Defines 1 iteration of train loop
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = nn.functional.mse_loss(x_hat, x)
    # Logging to TensorBoard by default
    self.log("train_loss", loss)
    return loss

  # Adam is an adaptive learning rate
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

autoencoder = LitAutoEncoder(encoder, decoder)

# Set up data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# Train the model
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# Load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder,
  decoder=decoder)

# Let's evaluate the encoder
# (Can also evaluate decoder)
encoder = autoencoder.encoder
encoder.eval()

# Embed 4 fake images!
fake_image_batch = Tensor(4, 28 * 28)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n",
  "⚡" * 20)
