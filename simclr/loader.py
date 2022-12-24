from torchvision import transforms
from torchvision.datasets import STL10

from const import DATASET_PATH

class Loader:
  def __init__(self):
    # To allow efficient training, apply random augmentations during data
    # loading

    # Apply set of 5 transformations
    self.contrast_transforms = transforms.Compose(
      [
        # Transformation 1: random horizontal flip
        transforms.RandomHorizontalFlip(),
        # Transformation 2: crop-and-resize
        transforms.RandomResizedCrop(size=96),
        # Transformation 3: colour distortion
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        # Transformation 4: random greyscale
        transforms.RandomGrayscale(p=0.2),
        # Transformation 5: Gaussian blur
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
      ]
    )
  
  def load_train(self):
    return STL10(
      root=DATASET_PATH,
      split="train",
      download=True,
      # transform takes in a function
      transform=ContrastiveTransformations(self.contrast_transforms, n_views=2),
    )
  
  def load_unlabeled(self):
    return STL10(
      root=DATASET_PATH,
      split="unlabeled",
      download=True,
      # transform takes in a function
      transform=ContrastiveTransformations(self.contrast_transforms, n_views=2),
    )

# ContrastiveTransformations is a class where instances behave like functions
# via defining __call__
class ContrastiveTransformations:
  def __init__(self, base_transforms, n_views=2):
    self.base_transforms = base_transforms
    self.n_views = n_views

  # https://www.geeksforgeeks.org/__call__-in-python/
  def __call__(self, x):
    return [self.base_transforms(x) for _ in range(self.n_views)]
