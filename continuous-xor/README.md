## Resources

- Section
  - Learning by example: Continuous XOR
  - https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/01-introduction-to-pytorch.html

## Run

### Dataset

`XORDataset` in `dataset.py` simulates creation of labelled data by creating a
vector ${(x, y)}$ where ${x, y \in \{ 0, 1 \}}$, labelling them accordingly with
XOR, then adding perturbation via Gaussian noise.

You can visualise the points by running the file.

```bash
python dataset.py
```

### Train

```bash
python train.py
```

### Test

```bash
python test.py
```
