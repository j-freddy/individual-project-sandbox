import torch
import torch.nn as nn
import torch.utils.data as data

from dataset import XORDataset
from net import SimpleClassifier
from utils import setup_gpu

# https://raw.githubusercontent.com/Lightning-AI/tutorials/main/course_UvA-DL/01-introduction-to-pytorch/small_neural_network.svg

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
  # Set model to train mode
  model.train()

  # Training loop
  # tqdm is a progress bar
  for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")

    for data_inputs, data_labels in data_loader:
      # Step 1: Move input data to device (only strictly necessary if we use GPU)
      data_inputs = data_inputs.to(device)
      data_labels = data_labels.to(device)

      # Step 2: Run the model on the input data
      preds = model(data_inputs)
      preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]

      # Step 3: Calculate the loss
      loss = loss_module(preds, data_labels.float())

      # Step 4: Perform backpropagation
      # Before calculating the gradients, we need to ensure that they are all zero.
      # The gradients would not be overwritten, but actually added to the existing ones.
      optimizer.zero_grad()
      # Perform backpropagation
      loss.backward()

      # Step 5: Update the parameters
      optimizer.step()

if __name__=="__main__":
  model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)

  # Use GPU if available
  device = setup_gpu()
  model.to(device)

  dataset = XORDataset(size=20)
  data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)

  # next(iter(...)) catches the first batch of the data loader
  # If shuffle is True, this will return a different batch every time we run
  # this cell
  # For iterating over the whole dataset, we can simple use "for batch in
  # data_loader: ..."
  data_inputs, data_labels = next(iter(data_loader))

  print("Data inputs", data_inputs.shape, "\n", data_inputs)
  print("Data labels", data_labels.shape, "\n", data_labels)

  # Binary Cross Entropy
  # Applied on 'logits' (logarithm) makes it more numerically stable
  loss_module = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

  # Train model
  # Let's use a larger dataset!
  train_dataset = XORDataset(size=1000)
  train_data_loader = data.DataLoader(train_dataset, batch_size=128,
    shuffle=True)

  train_model(model, optimizer, train_data_loader, loss_module)
  
  # Save model
  state_dict = model.state_dict()
  torch.save(state_dict, "model.tar")
