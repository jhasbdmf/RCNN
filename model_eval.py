import torch
from architecture import RCNN
from dataloader_test import ClutteredMNIST, get_mnist_cluttered_loaders

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)
          

            logits = model(images)[0]
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += B
    return correct / total * 100.0

train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
    root="./data", batch_size=256, val_fraction=0.1,
    image_size=64, n_clutter=50
)



# 1. Recreate the model
model = RCNN()
# 2. Load checkpoint safely to CPU, regardless of where it was saved
state_dict = torch.load("rcnn_neuroai_model_1.pth", map_location="cpu")

# 3. Load into model
model.load_state_dict(state_dict)

# 4. Optionally move model to the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


print (evaluate(model, test_loader))