import torch
from Dataset import get_dataset
from SimpleModel import DistractedDriverDetectionModel
from ResNet34 import ResNet34
from Train import train
from Utils import *


def main():
    # Device configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(device)

    # Model configuration
    model = DistractedDriverDetectionModel(input_size=3, num_classes=10)
    pretrained_model = ResNet34(num_classes=10)

    # model to the current device
    model = model.to(device)
    pretrained_model.to(device)

    # get data loaders and name of classes from Dataset.py
    train_dl, valid_dl, classes = get_dataset()
    print(classes)

    # Set parameters for simple model
    epochs = 1
    lr = 1e-4
    opt_func = torch.optim.RMSprop

    # Set parameters for ResNet34
    # epochs = 6
    # opt_func = torch.optim.Adam

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), lr)
    # optimizer = opt_func(pretrained_model.parameters(), lr)

    # start training
    history = train(epochs, model, train_dl, valid_dl, device, optimizer)
    # history = train(epochs, pretrained_model, train_dl, valid_dl, device, optimizer)


if __name__ == "__main__":
    main()
