import torch
import torchvision
import torchvision.transforms as transforms
import ssl

class Loader:
  def __init__(self, batch_size):
    # Turn off SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context

    self.batch_size = batch_size
    self.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  def load_train(self):
    trainset = torchvision.datasets.CIFAR10(
      root='./data',
      train=True,
      download=True,
      transform=self.transform
    )

    trainloader = torch.utils.data.DataLoader(
      trainset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

    return trainloader
  
  def load_test(self):
    testset = torchvision.datasets.CIFAR10(
      root='./data',
      train=False,
      download=True,
      transform=self.transform
    )

    testloader = torch.utils.data.DataLoader(
      testset,
      batch_size=self.batch_size,
      shuffle=False,
      # num_workers=0
      num_workers=2
    )

    return testloader
