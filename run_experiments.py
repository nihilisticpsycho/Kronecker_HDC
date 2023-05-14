import os
import tqdm


exp_configs = [
    # Dataset   D       epoch_hdc   epoch_metric    S_ou
    ['mnist',   1024,   15,         0,              8],
    ['face',    1024,   15,         0,              8],
    ['pamap2',  1024,   15,         0,              8],
    ['ucihar',  1024,   15,         0,              8],

    ['mnist',   1024,   1,         15,              8],
    ['face',    1024,   1,         15,              8],
    ['pamap2',  1024,   1,         15,              8],
    ['ucihar',  1024,   1,         15,              8],

    # ['isolet',  2048,   15,         0,              8],
    # ['mnist',   2048,   15,         0,              8],
    # ['face',    2048,   15,         0,              8],
    # ['pamap2',  2048,   15,         0,              8],
    # ['ucihar',  2048,   15,         0,              8],

    # ['isolet',  2048,   1,         15,              8],
    # ['mnist',   2048,   1,         15,              8],
    # ['face',    2048,   1,         15,              8],
    # ['pamap2',  2048,   1,         15,              8],
    # ['ucihar',  2048,   1,         15,              8],

    # ['isolet',  4096,   15,         0,              8],
    # ['mnist',   4096,   15,         0,              8],
    # ['face',    4096,   15,         0,              8],
    # ['pamap2',  4096,   15,         0,              8],
    # ['ucihar',  4096,   15,         0,              8],

    # ['isolet',  4096,   1,         15,              8],
    # ['mnist',   4096,   1,         15,              8],
    # ['face',    4096,   1,         15,              8],
    # ['pamap2',  4096,   1,         15,              8],
    # ['ucihar',  4096,   1,         15,              8],
]

for i in tqdm.tqdm(range(len(exp_configs))):
    log_path = './log_new/{}_{}_{}_{}_{}'.format(*exp_configs[i])
    os.mkdir(log_path)

    cmd = 'python main_RRAM_HD_margin.py -dataset {} -D {} -epoch_HDC {} -epoch_metric {} -S_ou {} -log_path {}'.format(*exp_configs[i], log_path)
    print(cmd)
    os.system(cmd)
    