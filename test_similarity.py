#%%
import NBHD
import dataset_utils
import torch

nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader\
    = dataset_utils.load_dataset("mnist", 256, "cuda")
model = NBHD.HDC(dim = nFeatures, D=1024, num_classes=nClasses, similarity_type="hamming")

m = 10
n = 5

random_hvs_1 = torch.torch.randint(-1,2, (m, 1024)).float().to("cuda")
random_hvs_2 = torch.torch.randint(0,2, (n, 1024)).float().to("cuda")
similarities = model.similarity(random_hvs_1, random_hvs_1)
print('similarity shape:', similarities.shape)
print('similarity matrix:', similarities)

# %%
