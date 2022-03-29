import torch
from tqdm import tqdm
import torch.nn as nn
from Utils import *


def train(epochs, model, train_dl, valid_dl, device, optimizer):
    torch.cuda.empty_cache()
    history = []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in tqdm(train_dl):
            images, labels = batch
            images = images.to(device)                      # images to the device
            labels = labels.to(device)                      # labels to the device
            out = model(images)                             # get predictions from model
            loss_fn = nn.CrossEntropyLoss()                 # loss function
            loss = loss_fn(out, labels)                     # calculate loss
            train_losses.append(loss)                       # store the iteration loss
            optimizer.zero_grad()
            loss.backward()                                 # calculate gradients
            optimizer.step()

        # Validation Phase
        result = evaluate(model, valid_dl, device)          # evaluate model using validation dataset
        result["train_loss"] = torch.stack(train_losses).mean().item()
        print("Epoch [{}/{}], train_loss : {:.4f}, val_loss : {:.4f}, val_acc : {:.4f}".format(epoch, epochs,
                                                                                               result["train_loss"],
                                                                                               result["val_loss"],
                                                                                               result["val_acc"]))
        history.append(result)                              # Store model accuracy, loss to evaluate later.

    return history
