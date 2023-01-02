import os

DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT",
  "saved-models/contrastive-learning/")

NUM_WORKERS = os.cpu_count()
