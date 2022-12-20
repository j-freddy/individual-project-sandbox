import torch
import torch.utils.data as data

from dataset import XORDataset
from net import SimpleClassifier
from utils import setup_gpu

def eval_model(model, data_loader):
  model.eval()  # Set model to eval mode

  true_preds, num_preds = 0.0, 0.0

  # Deactivate gradients for the following code
  with torch.no_grad():
    for data_inputs, data_labels in data_loader:
      # Determine prediction of model on dev set
      data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
      preds = model(data_inputs)
      preds = preds.squeeze(dim=1)
      preds = torch.sigmoid(preds)  # Sigmoid to map predictions between 0 and 1
      pred_labels = (preds >= 0.5).long()  # Binarize predictions to 0 and 1

      # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
      true_preds += (pred_labels == data_labels).sum()
      num_preds += data_labels.shape[0]

  acc = true_preds / num_preds
  print(f"Accuracy of the model: {100.0*acc:4.2f}%")

if __name__=="__main__":
  state_dict = torch.load("model.tar")

  # Create a new model and load the state
  model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
  model.load_state_dict(state_dict)

  # Use GPU if available
  device = setup_gpu()
  model.to(device)

  # Test data
  test_dataset = XORDataset(size=500)
  test_data_loader = data.DataLoader(test_dataset, batch_size=128,
    shuffle=False, drop_last=False)

  # Evaluate
  eval_model(model, test_data_loader)
