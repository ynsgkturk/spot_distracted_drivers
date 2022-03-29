import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

# Train path
train_path = "DistractedDriversDataset/train"
# Test path
test_path = "DistractedDriversDataset/test"

train_length = 0
for cls in os.listdir(train_path):
    print("%s size: %d" % (cls, len(os.listdir(os.path.join(train_path, cls)))))
    train_length += len(os.listdir(os.path.join(train_path, cls)))
print("Train size: %d" % train_length)
print("Test Size: %d" % len(os.listdir(test_path)))

# Data Transforms and Augmentation
train_transforms = T.Compose([T.Resize((64, 64)),
                              T.RandomAdjustSharpness(2),
                              T.RandomRotation((-15, 15)),
                              T.ColorJitter(brightness=.5, hue=.3),
                              T.ToTensor(),
                              ]
                             )

# Loading Data using ImageFolder
train_ds = ImageFolder(train_path, train_transforms)
classes = train_ds.classes


# Splitting into train-val set
val_pct = .1
val_size = int(val_pct * len(train_ds))
train_ds, valid_ds = random_split(train_ds, [len(train_ds)-val_size, val_size])


# Data Loader
batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size, num_workers=2, pin_memory=True)


def get_dataset():
    return train_dl, valid_dl, classes
