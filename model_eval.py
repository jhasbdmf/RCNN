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
            correct_per_batch = preds.eq(labels).sum().item()
            correct += correct_per_batch
            total += B
            print (f"{correct_per_batch} correct out of {B}")
    return correct / total * 100.0

train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
    root="./data", batch_size=256, val_fraction=0.1,
    image_size=64, n_clutter=50
)

def evaluate_rnn(model, loader):
    model.eval()
    correct = 0
    total = 0
    n_timesteps = 5
    timestep_correct = [n_timesteps]
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)
            total += B
          

            timestep_logits = model(images, n_timesteps)
            print (len(timestep_logits))
            
            for index, logits in enumerate(timestep_logits):
                _, preds = logits.max(1)
                correct_per_batch = preds.eq(labels).sum().item()
                correct += correct_per_batch
                timestep_correct[index] = correct
                print (f"{correct_per_batch} correct out of {B}")
    return timestep_correct / total * 100.0

train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
    root="./data", batch_size=256, val_fraction=0.1,
    image_size=64, n_clutter=50
)


"""
# 1. Recreate the model
ff_model = RCNN()
# 2. Load checkpoint safely to CPU, regardless of where it was saved
state_dict = torch.load("rcnn_neuroai_model_1.pth", map_location="cpu")

# 3. Load into model
ff_model.load_state_dict(state_dict)

# 4. Optionally move model to the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ff_model.to(device)


print (evaluate(ff_model, test_loader))
"""


# 1. Recreate the model
r_model = RCNN()
# 2. Load checkpoint safely to CPU, regardless of where it was saved
state_dict = torch.load("rcnn_neuroai_model_5.pth", map_location="cpu")

# 3. Load into model
r_model.load_state_dict(state_dict)

# 4. Optionally move model to the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r_model.to(device)


print ("Accs of rnn: ", evaluate_rnn(r_model, test_loader))