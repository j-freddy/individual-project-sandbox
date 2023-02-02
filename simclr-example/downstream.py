from copy import deepcopy
import os
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.datasets import STL10
import torchvision.transforms as transforms

from const import CHECKPOINT_PATH, DATASET_PATH, NUM_WORKERS
from logistic_regression import LogisticRegression
from simclr import SimCLR
from utils import get_smaller_dataset, setup_gpu

# "Next, we implement a small function to encode all images in our datasets. The
# output representations are then used as inputs to the Logistic Regression
# model."
@torch.no_grad()
def prepare_data_features(model, dataset):
  # Prepare model
  network = deepcopy(model.convnet)
  network.fc = nn.Identity()  # Removing projection head g(.)
  network.eval()
  network.to(device)

  # Encode all images
  data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
  feats, labels = [], []
  for batch_imgs, batch_labels in data_loader:
    batch_imgs = batch_imgs.to(device)
    batch_feats = network(batch_imgs)
    feats.append(batch_feats.detach().cpu())
    labels.append(batch_labels)

  feats = torch.cat(feats, dim=0)
  labels = torch.cat(labels, dim=0)

  # Sort images by labels
  labels, idxs = labels.sort()
  feats = feats[idxs]

  return data.TensorDataset(feats, labels)

# ==============================================================================
# Training logistic regression
# ==============================================================================
def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
  trainer = pl.Trainer(
    default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
    gpus=1 if str(device) == "cuda:0" else 0,
    max_epochs=max_epochs,
    callbacks=[
      ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
      LearningRateMonitor("epoch"),
    ],
    check_val_every_n_epoch=10,
  )
  trainer.logger._default_hp_metric = None

  # Data loaders
  train_loader = data.DataLoader(
    train_feats_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=0
  )
  test_loader = data.DataLoader(
    test_feats_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0
  )

  # Check whether pretrained model exists. If yes, load it and skip training
  pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
  if os.path.isfile(pretrained_filename):
    print(f"Found pretrained model at {pretrained_filename}, loading...")
    model = LogisticRegression.load_from_checkpoint(pretrained_filename)
  else:
    pl.seed_everything(42)  # To be reproducable
    model = LogisticRegression(**kwargs)
    trainer.fit(model, train_loader, test_loader)
    model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

  # Test best model on train and validation set
  train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
  test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
  result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

  return model, result

if __name__=="__main__":
  device = setup_gpu()

  img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  train_img_data = STL10(
    root=DATASET_PATH,
    split="train",
    download=True,
    transform=img_transforms
  )
  
  test_img_data = STL10(
    root=DATASET_PATH,
    split="test",
    download=True,
    transform=img_transforms
  )

  print("Number of training examples:", len(train_img_data))
  print("Number of test examples:", len(test_img_data))

  # Get pretrained model
  # pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLRFreddy.ckpt")
  pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")

  if os.path.isfile(pretrained_filename):
    print(f"Found pretrained model at {pretrained_filename}, loading...")
    # Automatically loads the model with saved hyperparameters
    simclr_model = SimCLR.load_from_checkpoint(pretrained_filename)
  else:
    raise Exception("No model found.")

  print("Preparing data features...")

  train_feats_simclr = prepare_data_features(simclr_model, train_img_data)
  test_feats_simclr = prepare_data_features(simclr_model, test_img_data)

  print("Preparing data features: done!")

  # Train downstream model
  print("Train downstream models...")
  results = {}

  for num_imgs_per_label in [10, 20, 50]:
    print(f"Training: {num_imgs_per_label}")
    sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)
    _, small_set_results = train_logreg(
      batch_size=64,
      train_feats_data=sub_train_set,
      test_feats_data=test_feats_simclr,
      model_suffix=num_imgs_per_label,
      feature_dim=train_feats_simclr.tensors[0].shape[1],
      num_classes=10,
      lr=1e-3,
      weight_decay=1e-3,
    )
    results[num_imgs_per_label] = small_set_results
  
  # Plot results
  dataset_sizes = sorted(k for k in results)
  test_scores = [results[k]["test"] for k in dataset_sizes]

  fig = plt.figure(figsize=(6, 4))
  plt.plot(
    dataset_sizes,
    test_scores,
    "--",
    color="#000",
    marker="*",
    markeredgecolor="#000",
    markerfacecolor="y",
    markersize=16,
  )
  plt.xscale("log")
  plt.xticks(dataset_sizes, labels=dataset_sizes)
  plt.title("STL10 classification over dataset size", fontsize=14)
  plt.xlabel("Number of images per class")
  plt.ylabel("Test accuracy")
  plt.minorticks_off()
  plt.show()

  for k, score in zip(dataset_sizes, test_scores):
    print(f"Test accuracy for {k:3d} images per label: {100*score:4.2f}%")
