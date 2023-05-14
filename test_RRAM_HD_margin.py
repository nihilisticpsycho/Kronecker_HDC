# %%
from RRAM import RRAM
import HD

import numpy as np

import torch
import torch.nn as nn

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from torchvision import datasets, transforms
import torch.optim as optim


def test_RRAM_HD(model, x_test, y_test):
    class_hvs = HD.binarize_hard(model.class_hvs)
    test_hvs = HD.binarize_hard(model(x_test, embedding=True))

    preds = torch.argmax(test_hvs@class_hvs.T, dim=-1)
    fp_acc = torch.mean((preds==y_test).float()).item()
    print("Full-precision acc:", fp_acc)

    unipolar_class_hvs = torch.where(class_hvs>0, 1, 0).cpu().numpy()
    unipolar_test_hvs = torch.where(test_hvs>0, 1, 0).cpu().numpy()

    rram_chip = RRAM()
    print("S_ou = {}".format(rram_chip.S_ou))
    avg_loss_list = []
    for i in range(5):
        rram_chip.rram_write_binary(unipolar_class_hvs)

        preds, Hamming_sim_cim = rram_chip.rram_hd_am(unipolar_test_hvs, collect_stats=True)

        acc = torch.mean((torch.tensor(preds).cuda() == y_test).float()).item()
        acc_loss = acc-fp_acc
        avg_loss_list.append(acc_loss)
        print("Test accuracy = {:.4f} ({:.4f})".format(acc, acc_loss))
    print("Mean Acc loss = {:.4f}".format(np.mean(avg_loss_list)))



# %%

device = "cuda"
transform = transforms.Compose(
    [
		transforms.ToTensor(), 
		# transforms.Normalize((0.1307,), (0.3081,)),
		transforms.Lambda(lambda x: torch.flatten(x))
	]
)

batch_size = 256

dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST(".", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


model = HD.HD_RP(input_size=int(28*28), D=1024, num_classes=10, binary=True)

x_train = torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1), shuffle=True)
for i in x_train:
	x_train, y_train = i
	x_train, y_train = x_train.to(device), y_train.to(device)

x_test = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2), shuffle=False)
for i in x_test:
	x_test, y_test = i
	x_test, y_test = x_test.to(device), y_test.to(device)


model.init_class(x_train, y_train)

train_epoch = 3
for i in range(train_epoch):
    print("HD training epoch: ", i)
    model.HD_train_step(x_train, y_train)
    HD.HD_test(model, x_test, y_test)
    HD.get_Hamming_margin(model, x_test)
    test_RRAM_HD(model, x_test, y_test)

model.class_hvs = nn.parameter.Parameter(data=model.class_hvs)


num_epochs = 5

optimizer = optim.Adam(model.parameters(), lr=0.01)

### pytorch-metric-learning stuff ###
# distance = distances.CosineSimilarity()
distance = distances.LpDistance()


reducer = reducers.ThresholdReducer(low=0.0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard")
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

### pytorch-metric-learning stuff ###
for epoch in range(1, num_epochs + 1):
    print("Metric learning epoch: ", epoch)
    HD.train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    HD.test(dataset1, dataset2, model, accuracy_calculator)
    HD.HD_test(model, x_test, y_test)
    HD.get_Hamming_margin(model, x_test)
    test_RRAM_HD(model, x_test, y_test)

exit()

# %% 1. Run HD training
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = np.array(train_images).reshape(60000,-1)
test_images = np.array(test_images).reshape(10000,-1)

D = 1024
train_epoch = 20
if_restore = True


if if_restore:
    class_hvs_binary = np.load('class_hvs_binary.pkl', allow_pickle=True)
    enc_base_mat_binary = np.load('enc_base_mat_binary.pkl', allow_pickle=True)
else:
    acc_binary, class_hvs_binary, enc_base_mat_binary = HD.train_binary(
        X_train=train_images, y_train=train_labels,
        X_test=test_images, y_test=test_labels,
        D=D, alg = 'rp-Kron', Kron_shape=[32, 28], epoch=train_epoch, lr=1.0)

    class_hvs_binary.dump('class_hvs_binary.pkl')
    enc_base_mat_binary = np.array(enc_base_mat_binary)
    enc_base_mat_binary.dump('enc_base_mat_binary.pkl')


# %% 1. Full-precision accuracy test
print("1. Full-precision accuracy test:")
HD.run_test(
    class_hvs=class_hvs_binary, 
    X_test=test_images, y_test=test_labels, 
    base_matrix=enc_base_mat_binary, alg='rp-Kron')

# class_Hamming_distance = class_hvs_binary @ class_hvs_binary.T
# class_Hamming_distance = (D-class_Hamming_distance)/2
# print(class_Hamming_distance)

unipolar_class_hvs = np.where(class_hvs_binary>0, 1, 0)
test_enc_hvs = HD.encoding_Kron(test_images, enc_base_mat_binary, signed=True)
test_enc_hvs = np.array(test_enc_hvs)
unipolar_test_hvs = np.where(test_enc_hvs>0, 1, 0)


# %% 2. Test Accuracy on ReRAM
WL_act = 4
R_sigma = [0.18, 0.45]
# R_sigma = [0.36, 0.9]

rram_chip = RRAM(R_deviation=R_sigma,S_ou=WL_act)
rram_chip.rram_write_binary(unipolar_class_hvs)
rram_chip.plot_rram_cell_stats()

# 2.1 Test HD AM search near the ReRAM chip
print("\n2. Test HD AM search near the ReRAM chip")
read_class_hvs = rram_chip.rram_read_binary()
error_rate = np.sum(np.abs(read_class_hvs-unipolar_class_hvs))/unipolar_class_hvs.size
print("Error rate when reading from ReRAM: {:.4f}".format(error_rate))

test_acc = HD.run_test(
    class_hvs=read_class_hvs, 
    X_test=test_images, y_test=test_labels, 
    base_matrix=enc_base_mat_binary, alg='rp-Kron')


# 2.2 Test HD AM search using ReRAM-based CIM
 
print("\n3. Test HD AM search using CIM")
rram_chip.plot_rram_cim_stats()


for i in range(5):
    preds, Hamming_sim_cim = rram_chip.rram_hd_am(unipolar_test_hvs, collect_stats=True)

    acc = np.mean(preds == test_labels)
    print("S_ou = {}".format(WL_act))
    print("Test accuracy = {:.4f}".format(acc))

    rram_chip.rram_write_binary(unipolar_class_hvs)


def get_Hamming_margin(x):
    sorted_x = np.sort(x, axis=-1)
    return sorted_x[:,-1] - sorted_x[:, -2]
    
def get_in_class_max_error(x):
    sorted_x = np.sort(x, axis=-1)
    return sorted_x[:,-1] - sorted_x[:, 0]

# %% Analyze the robustness of HD
def estimate_cim_Hamming_distance_error(test_hvs, class_hvs, error_table):
    N, C, D = test_hvs.shape[0], class_hvs.shape[0], class_hvs.shape[1]
    WL_act = error_table.shape[0]-1

    # Step 1
    Hamming_match_1 = test_hvs[:, np.newaxis, :]*class_hvs # N, C, D
    Hamming_match_1 = Hamming_match_1.reshape(N, C, -1, WL_act).sum(axis=-1) # N, C, D//S_ou
    
    Hamming_match_1_exp = (error_table[Hamming_match_1.flatten(), :] * np.arange(WL_act+1)).reshape(N, C, D//WL_act, WL_act+1) # N, C, D//S_ou, WL_act+1
    Hamming_match_1_exp = Hamming_match_1_exp.reshape(N, C, -1).sum(axis=-1) # N, C

    # Step 2
    Hamming_match_01 = (1-test_hvs[:, np.newaxis, :])*class_hvs # N, C, D
    Hamming_match_01 = Hamming_match_01.reshape(test_hvs.shape[0], class_hvs.shape[0], -1, WL_act).sum(axis=-1) # N, C, D//S_ou
    
    Hamming_match_01_exp = (error_table[Hamming_match_01.flatten(), :] * np.arange(WL_act+1)).reshape(N, C, D//WL_act, WL_act+1) # N, C, D//S_ou, WL_act+1
    Hamming_match_01_exp = Hamming_match_01_exp.reshape(N, C, -1).sum(axis=-1) # N, C

    zero_cnt_query = (1-test_hvs).sum(axis=-1).reshape(N, 1) # N, 1
    Hamming_match_0_exp = zero_cnt_query - Hamming_match_01_exp

    # 
    Hamming_match_1_correct = test_hvs[:, np.newaxis, :]*class_hvs # N, C, D
    Hamming_match_0_correct = (1-test_hvs)[:, np.newaxis, :]*(1-class_hvs) # N, C, D

    plt.figure()
    diff = Hamming_match_1_correct.sum(axis=-1) - Hamming_match_1_exp
    sns.distplot(diff, label='Matched 1 error')
    diff = Hamming_match_0_correct.sum(axis=-1) - Hamming_match_0_exp
    sns.distplot(diff, label='Matched 0 error')
    plt.show()

    Hamming_xor = (test_hvs[:, np.newaxis, :]-class_hvs)==0
    Hamming_sim_correct = Hamming_xor.reshape(N, C, -1).sum(axis=-1)
    Hamming_sim_exp = Hamming_match_0_exp + Hamming_match_1_exp # N, C
    diff_Hamming = Hamming_sim_correct-Hamming_sim_exp

    print(np.abs(diff_Hamming).mean())

    
rram_chip.build_error_table()

print("CIM error table: \n", rram_chip.error_table)

estimate_cim_Hamming_distance_error(unipolar_test_hvs, unipolar_class_hvs, rram_chip.error_table)



# %% Plot auxiliary metrics
Hamming_xor = (unipolar_test_hvs[:, np.newaxis, :]-unipolar_class_hvs)==0

gt_Hamming_sim_seg = Hamming_xor.reshape(unipolar_test_hvs.shape[0], unipolar_class_hvs.shape[0], -1, WL_act).sum(axis=-1) # N, C, D//S_ou

diff_Hamming_sim_seg = gt_Hamming_sim_seg - Hamming_sim_cim

plt.figure()
sns.distplot(diff_Hamming_sim_seg[0].flatten())
plt.title("Hamming Seg CIM error distribution")
plt.xlabel('Difference')
plt.show()


diff_Hamming = gt_Hamming_sim_seg.sum(axis=-1) - Hamming_sim_cim.sum(axis=-1)

plt.figure()
sns.distplot(diff_Hamming.flatten())
plt.title("Hamming CIM error distribution")
plt.xlabel('Difference')
plt.show()


zero_cnt_query = (unipolar_test_hvs==0).reshape(unipolar_test_hvs.shape[0], -1, WL_act).sum(axis=-1)

plt.figure()
sns.distplot(zero_cnt_query)
plt.title("Distribution of zero_cnt_query")
plt.xlabel('Zero CNT')
plt.show()


gt_Hamming_sim = gt_Hamming_sim_seg.sum(axis=-1)
Hamming_margin = get_Hamming_margin(gt_Hamming_sim)

plt.figure()
sns.distplot(Hamming_margin)
plt.title("Hamming margin")
plt.xlabel('Margin')
plt.show()


diff_Hamming_sim = gt_Hamming_sim - Hamming_sim_cim.sum(axis=-1)
max_in_class_Hamming_sim_error = get_in_class_max_error(diff_Hamming_sim)

plt.figure()
sns.distplot(max_in_class_Hamming_sim_error)
plt.title("Max in-class Hamming error")
plt.xlabel('Error')
plt.show()


overflow = Hamming_margin - max_in_class_Hamming_sim_error

plt.figure()
sns.distplot(overflow)
plt.title("Diff. between Hamming margin and max in-class Hamming error")
plt.xlabel('Difference')
plt.axvline(0, color='r', ls='--')
plt.show()


