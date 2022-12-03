import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import cnn
import loader
import visualiser

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

batch_size = 4

trainloader = loader.load(batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse'
  'ship', 'truck')

if __name__=="__main__":
  # # Get some random training images
  # dataiter = iter(trainloader)
  # images, labels = next(dataiter)

  # # Print labels
  # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
  # # Show images
  # visualiser.imshow(torchvision.utils.make_grid(images))

  net = cnn.Net()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
      inputs, labels = data

      optimizer.zero_grad()

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      
      # Print every 2000 mini-batches
      if i % 2000 == 1999:
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

  print('Finished Training')

  PATH = './cifar_net.pth'
  torch.save(net.state_dict(), PATH)
