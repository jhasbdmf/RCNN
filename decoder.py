from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from architecture import RCNN
from dataloader_test import ClutteredMNIST, get_mnist_cluttered_loaders


import torch

def apply_pca_to_batch(tensors, n_components=128):
    tensor_cpu = tensors.to('cpu')
    np_array = tensor_cpu.detach().numpy()
    flat_array = np_array.reshape(np_array.shape[0], -1)  # batch size x features
    pca = PCA(n_components=n_components)
    return pca.fit_transform(flat_array)



logistic_regression = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",   # or "saga" for large / sparse data
    C=1.0,
    max_iter=1000,
    random_state=0,
)

train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
    root="./data", batch_size=256, val_fraction=0.1,
    image_size=64, n_clutter=50
)



# 1. Recreate the model
r_model = RCNN()
# 2. Load checkpoint safely to CPU, regardless of where it was saved
state_dict = torch.load("rcnn_neuroai_model_5.pth", map_location="cpu")

# 3. Load into model
r_model.load_state_dict(state_dict)

# 4. Optionally move model to the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r_model.to(device)

for inputs, labels, _ in test_loader:
    activations = r_model(inputs, return_actvs=True)
    for key, value in activations.items():

        print (key)
        for layer_act in value:
            print (type(value[0]), layer_act.shape)
            print ("_"*5)
            print (type(value[0]), len(apply_pca_to_batch(layer_act)))
            print ("_"*10)
    break