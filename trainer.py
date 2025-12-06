from dataloader_test import ClutteredMNIST, get_mnist_cluttered_loaders
from architecture import RCNN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
        root="./data", batch_size=16, val_fraction=0.1,
        image_size=64, n_clutter=50
    )

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

    abc = RCNN()