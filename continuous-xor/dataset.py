from matplotlib import pyplot as plt
import torch
import torch.utils.data as data

# Define 2 functions:
#   __getitem__
#   __len__

class XORDataset(data.Dataset):
  def __init__(self, size, std=0.1):
    """
    Inputs:
        size - Number of data points we want to generate
        std - Standard deviation of the noise (see generate_continuous_xor function)
    """
    super().__init__()
    self.size = size
    self.std = std
    # Simulate creation of data
    self.generate_continuous_xor()

  def generate_continuous_xor(self):
    # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
    # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
    # If x=y, the label is 0.
    data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
    label = (data.sum(dim=1) == 1).to(torch.long)
    # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
    data += self.std * torch.randn(data.shape)

    self.data = data
    self.label = label

  def __len__(self):
    # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
    return self.size

  def __getitem__(self, idx):
    # Return the idx-th data point of the dataset
    # If we have multiple things to return (data point and label), we can return them as tuple
    data_point = self.data[idx]
    data_label = self.label[idx]
    return data_point, data_label

def visualize_samples(data, label):
  if isinstance(data, torch.Tensor):
      data = data.cpu().numpy()
  if isinstance(label, torch.Tensor):
      label = label.cpu().numpy()
  data_0 = data[label == 0]
  data_1 = data[label == 1]

  plt.figure(figsize=(4, 4))
  plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
  plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
  plt.title("Dataset samples")
  plt.ylabel(r"$x_2$")
  plt.xlabel(r"$x_1$")
  plt.legend()

if __name__=="__main__":
  dataset = XORDataset(size=200)
  visualize_samples(dataset.data, dataset.label)
  plt.show()

