import torch
import torchvision

from const import classes, PATH
import cnn
from loader import Loader
import visualiser

if __name__=="__main__":
  loader = Loader(batch_size=4)
  testloader = loader.load_test()

  dataiter = iter(testloader)
  images, labels = next(dataiter)

  # # Show example images
  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
  # visualiser.imshow(torchvision.utils.make_grid(images))

  net = cnn.Net()
  # Load saved trained NN model
  net.load_state_dict(torch.load(PATH))

  # What the NN model thinks the test images are
  # @outputs are energies for the 10 classes
  outputs = net(images)

  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    for j in range(4)))

  # Test how network performs on whole dataset

  correct = 0
  total = 0

  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)

      # Track accuracy for whole dataset
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f'Accuracy of the network on the 10000 test images:\
    {100 * correct // total} %')
