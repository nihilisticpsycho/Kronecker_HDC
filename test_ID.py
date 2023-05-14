#%%
import NBHD
import dataset_utils
import torch

nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader\
        = dataset_utils.load_dataset('pamap2', 256, "cpu")
model = NBHD.HDC(dim = nFeatures, D=1000, m=5, enc_type='ID', num_classes=nClasses, levels=3, similarity_type= 'cosine')
model.init_class(x_train, y_train)
print('class_hvs:', model.class_hvs)
print('level_hvs:',model.level_hvs)
print('base_hvs:', model.base_hvs)

print('base_hvs similarity:', model.similarity(model.base_hvs, model.base_hvs))
print('level_hvs similarity:', model.similarity(model.level_hvs, model.level_hvs))

print('encoding test:', model.encoding(x_train, quantize=False))

# %%
