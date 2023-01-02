import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

class SimCLR(pl.LightningModule):
  def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
    super().__init__()

    # https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html
    # save_hyperparameters() saves all __init__ parameters into hparams:
    #   self.hparams.hidden_dim
    #   self.hparams.lr
    #   self.hparams.temperature
    #   self.hparams.weight_decay
    #   self.hparams.max_epochs
    self.save_hyperparameters()

    assert self.hparams.temperature > 0.0

    # See simclr.png
    # f(.) is a CNN and g(.) is MLP (multi-layer perceptron)

    # https://raw.githubusercontent.com/Lightning-AI/tutorials/main/course_UvA-DL/13-contrastive-learning/simclr_network_setup.svg

    # f(.) - we use ResNet which is good for computer vision
    # num_classes is output size of the last linear layer
    self.convnet = torchvision.models.resnet18(pretrained=False,
      num_classes=4 * hidden_dim)

    # MLP for g(.) consists of Linear->ReLU->Linear
    # (as mentioned in the 2nd paper in my individual project description)
    # ResNet -> FC Layer –{ReLU}> FC Layer -> Loss
    self.convnet.fc = nn.Sequential(self.convnet.fc, nn.ReLU(inplace=True),
      nn.Linear(4 * hidden_dim, hidden_dim))

  def configure_optimizers(self):
    # AdamW decouples weight decay from gradient updates
    optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
      weight_decay=self.hparams.weight_decay)

    # Set learning rate using a cosine annealing schedule
    # See https://pytorch.org/docs/stable/optim.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
      T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)

    return [optimizer], [lr_scheduler]

  def info_nce_loss(self, batch, mode="train"):
    imgs, _ = batch
    # Concatenates tensors into 1D
    imgs = torch.cat(imgs, dim=0)

    # Encode all images using ResNet - this is f(.) in the diagram
    # "On those images, we apply a CNN like ResNet and obtain as output a 1D
    # feature vector on which we apply a small MLP."
    feats = self.convnet(imgs)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool,
      device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    # InfoNCE loss
    cos_sim /= self.hparams.temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Logging loss
    self.log(mode + "_loss", nll)

    # Get ranking position of positive example
    comb_sim = torch.cat(
      [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
      # First position positive example
      dim=-1,
    )

    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

    # Logging ranking metrics
    self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
    self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
    self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

    return nll

  # Training step returns loss
  def training_step(self, batch, batch_idx):
    return self.info_nce_loss(batch, mode="train")

  # Validation step returns nothing
  def validation_step(self, batch, batch_idx):
    self.info_nce_loss(batch, mode="val")
