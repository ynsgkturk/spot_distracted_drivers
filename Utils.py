import torch
import torch.nn.functional as F


# Accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Evaluate function
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = []
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        outputs.append({"val_loss": loss.detach(), "val_acc": acc})

    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine Losses
    batch_accs = [x["val_acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine Accuracies
    return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
