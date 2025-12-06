from dataloader_test import ClutteredMNIST, get_mnist_cluttered_loaders
from architecture import RCNN
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch
from datetime import datetime


def train_model (model, train_loader, val_loader, n_epochs, device):

    model.train()


    train_loss_history = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, _ in train_loader:
            inputs = inputs.to (device)
            labels = labels.to (device)

            logits = model(inputs)[-1]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)

            running_loss += loss.item() * batch_size
            _, preds = logits.max(1)
            correct_per_batch = preds.eq(labels).sum().item()
            correct += correct_per_batch
            total += batch_size
            print (f"batch acc = {correct_per_batch/batch_size}, batch_size = {batch_size}")

        avg_loss = running_loss / total
        train_loss_history.append (avg_loss)
        acc = correct / total * 100.0
        #print(f"Task {task_id} | Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    return model, train_loss_history




if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
        root="./data", batch_size=16, val_fraction=0.1,
        image_size=64, n_clutter=50
    )


    """
    imgs, labels, centers = next(iter(train_loader))
    print(imgs.shape, labels.shape, centers.shape)

    n_show = 8
    plt.figure(figsize=(12, 4))
    for i in range(n_show):
        img = imgs[i, 0].numpy()
        label = labels[i].item()
        cx_norm, cy_norm = centers[i].tolist()
        H, W = img.shape
        cx, cy = cx_norm * (W - 1), cy_norm * (H - 1)

        ax = plt.subplot(2, n_show // 2, i + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.scatter([cx], [cy], c="red", s=10)
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    """
    abc = RCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    abc_trained, train_loss_hisotry = train_model(model=abc, train_loader=train_loader, val_loader=val_loader, n_epochs=2, device = device)


    print (train_loss_hisotry)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20251206_130945
    filename = f"trained_model_{timestamp}.pth"



    torch.save(abc_trained.state_dict(), filename)