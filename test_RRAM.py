# %%
from RRAM import RRAM

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
import pandas as pd

# %% Test ReRAM memory functions
WL_act = 4
test = RRAM(S_ou=WL_act)

test.plot_rram_cell_stats()

data = np.random.randint(2, size=(1024,1024))
test.rram_write_binary(data)

read_data = test.rram_read_binary()
print("Error rate:", np.sum(np.abs(read_data-data))/data.size)


# %% Test TSMC's RRAM parameters
mu = 2500
sigma = 0.18
Ron_dist = test.init_distribution(mu, sigma)

mu = 16000
sigma = 0.45
Roff_dist = test.init_distribution(mu, sigma)

print("99% range of Ron: ", Ron_dist.interval(0.99))
print("99% range of Ron: ", Roff_dist.interval(0.99))


# %%
# 1. Resistance distribution
plt.figure()
sns.distplot(test.Ron_dist.rvs(10000))
sns.distplot(test.Roff_dist.rvs(10000))
plt.title("Resistance distribution")

# 2. Actual current distribution
Ron = test.Ron_dist.rvs(10000)
Roff = test.Roff_dist.rvs(10000)
read_current_on = test.Vread/Ron
read_current_off = test.Vread/Roff

plt.figure()
sns.distplot(read_current_off)
sns.distplot(read_current_on)
plt.title('Actual readout current distribution')

# 3. Fitted current distribution based on lognorm
Ion_fit_parm = stats.lognorm.fit(read_current_on, floc=0)
Ioff_fit_parm = stats.lognorm.fit(read_current_off, floc=0)

Ion_fit_lognorm = stats.lognorm(*Ion_fit_parm)
Ioff_fit_lognorm = stats.lognorm(*Ioff_fit_parm)

Ion_rvs = Ion_fit_lognorm.rvs(10000)
Ioff_rvs = Ioff_fit_lognorm.rvs(10000)

plt.figure()
sns.distplot(Ion_rvs)
sns.distplot(Ioff_rvs)
plt.title('Fitted lognorm readout current')

# 4. Theoretical readout current using lognorm
Ion_lognorm = stats.lognorm(s=0.18, scale=test.Vread/test.Ron)
Ioff_lognorm = stats.lognorm(s=0.45, scale=test.Vread/test.Roff)

plt.figure()
sns.distplot(Ion_lognorm.rvs(10000))
sns.distplot(Ioff_lognorm.rvs(10000))
plt.title('Lognorm current')


# %% Test ReRAM CIM functions
# Test ReRAM CIM error
WL_act = 8
test = RRAM(S_ou=WL_act)

inp = np.random.randint(2, size=1000000)
weight = np.random.randint(2, size=1000000)
test.rram_write_binary(weight)

BL_current = test.Vread*inp/test.rram_array
BL_acc_current = BL_current.reshape(-1, WL_act).sum(axis=-1)

test.plot_rram_cim_stats()
sns.distplot(BL_acc_current, label='BL_acc_current')
plt.legend()

gt_compute_out = (inp*weight).reshape(-1, WL_act).sum(axis=-1)
sense_out = test.rram_sense_current_cim(BL_acc_current.flatten()).reshape(BL_acc_current.shape)
diff = gt_compute_out-sense_out


comp_digits = np.arange(WL_act+1)
cim_conf_mat = confusion_matrix(gt_compute_out, sense_out, labels=comp_digits)

df_cm = pd.DataFrame(cim_conf_mat, comp_digits, comp_digits)
plt.figure(figsize=(12,12))
sns.heatmap(df_cm, annot=True) 

# sns.distplot(diff, label='BL_acc_current')

# %%
