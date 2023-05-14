import os
import tqdm


exp_configs = [
    # Dataset   D       R                   R_deviation     S_ou  
    ['mnist',    1024,   [2500, 16000],      [0.18, 0.45],   4],
    ['face',     1024,   [2500, 16000],      [0.18, 0.45],   4],
    ['pamap2',   1024,   [2500, 16000],      [0.18, 0.45],   4],
    ['ucihar',   1024,   [2500, 16000],      [0.18, 0.45],   4],

    ['mnist',    1024,   [2500, 16000],      [0.18, 0.45],   8],
    ['face',     1024,   [2500, 16000],      [0.18, 0.45],   8],
    ['pamap2',   1024,   [2500, 16000],      [0.18, 0.45],   8],
    ['ucihar',   1024,   [2500, 16000],      [0.18, 0.45],   8],

    ['mnist',    1024,   [2500, 16000],      [0.18, 0.45],   16],
    ['face',     1024,   [2500, 16000],      [0.18, 0.45],   16],
    ['pamap2',   1024,   [2500, 16000],      [0.18, 0.45],   16],
    ['ucihar',   1024,   [2500, 16000],      [0.18, 0.45],   16],

    ['mnist',    1024,   [2500, 16000],      [0.18*2, 0.45*2],   8],
    ['face',     1024,   [2500, 16000],      [0.18*2, 0.45*2],   8],
    ['pamap2',   1024,   [2500, 16000],      [0.18*2, 0.45*2],   8],
    ['ucihar',   1024,   [2500, 16000],      [0.18*2, 0.45*2],   8],

    ['mnist',    1024,   [2500, 16000],      [0.18*3, 0.45*3],   8],
    ['face',     1024,   [2500, 16000],      [0.18*3, 0.45*3],   8],
    ['pamap2',   1024,   [2500, 16000],      [0.18*3, 0.45*3],   8],
    ['ucihar',   1024,   [2500, 16000],      [0.18*3, 0.45*3],   8],
]


metric_model_path_dict = {
    'mnist': './log_new/mnist_1024_1_15_8/model_metric_epoch_15.ckpt',
    'face': './log_new/face_1024_1_15_8/model_metric_epoch_15.ckpt',
    'pamap2': './log_new/pamap2_1024_1_15_8/model_metric_epoch_15.ckpt',
    'ucihar': './log_new/ucihar_1024_1_15_8/model_metric_epoch_15.ckpt',
}

hydrea_model_path_dict = {
    'mnist': './log_new/mnist_1024_15_0_8/model_hdc_epoch_15.ckpt',
    'face': './log_new/face_1024_15_0_8/model_hdc_epoch_15.ckpt',
    'pamap2': './log_new/pamap2_1024_15_0_8/model_hdc_epoch_15.ckpt',
    'ucihar': './log_new/ucihar_1024_15_0_8/model_hdc_epoch_11.ckpt',
}

metric_class_path_dict = {
    'mnist': './log_new/mnist_1024_1_15_8/model_metric_class_hvs_15.pt',
    'face': './log_new/face_1024_1_15_8/model_metric_class_hvs_15.pt',
    'pamap2': './log_new/pamap2_1024_1_15_8/model_metric_class_hvs_15.pt',
    'ucihar': './log_new/ucihar_1024_1_15_8/model_metric_class_hvs_15.pt',
}

hydrea_class_path_dict = {
    'mnist': './log_new/mnist_1024_15_0_8/model_hdc_class_hvs_15.pt',
    'face': './log_new/face_1024_15_0_8/model_hdc_class_hvs_15.pt',
    'pamap2': './log_new/pamap2_1024_15_0_8/model_hdc_class_hvs_15.pt',
    'ucihar': './log_new/ucihar_1024_15_0_8/model_hdc_class_hvs_11.pt',
}


for i in tqdm.tqdm(range(len(exp_configs))):
    for alg in ['hydrea', 'hypermetric']:
        if alg == 'hydrea':
            model_path = hydrea_model_path_dict[exp_configs[i][0]]
            class_path = hydrea_class_path_dict[exp_configs[i][0]]
        else:
            model_path = metric_model_path_dict[exp_configs[i][0]]
            class_path = metric_class_path_dict[exp_configs[i][0]]
    
        log_path = './log_rram'

        cmd = 'python test_single_model.py -algorithm {} -dataset {} -D {} -R {} {} -R_deviation {} {} -S_ou {} -model_path {} -class_path {} -log_path {}'.format(
            alg, exp_configs[i][0], exp_configs[i][1], *exp_configs[i][2], *exp_configs[i][3], exp_configs[i][4], model_path, class_path, log_path)
        print(cmd)
        os.system(cmd)
    