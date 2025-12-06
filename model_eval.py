import torch
from architecture import RCNN
from dataloader_test import ClutteredMNIST, get_mnist_cluttered_loaders

# 1. Recreate the model
model = RCNN()
# 2. Load checkpoint safely to CPU, regardless of where it was saved
state_dict = torch.load("rcnn_neuroai_model_1.pth", map_location="cpu")

# 3. Load into model
model.load_state_dict(state_dict)

# 4. Optionally move model to the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()