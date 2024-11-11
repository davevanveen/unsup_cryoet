import os

############################################################
### set user, system-specific paths here ###################

USER = 'dvv'

if USER == 'dvv':
    #DIR_PROJECT = '/home/vanveen/coord_cryo_et/logs'
    #DIR_DATA = '/data/dvv/cryo/'
    DIR_PROJECT = '/home/vanveen/wucon_test'
    DIR_DATA = os.path.join(DIR_PROJECT, 'data') 
elif USER == 'you':
    DIR_PROJECT = '/your/path/to/wucon/project'
    DIR_DATA = os.path.join(DIR_PROJECT, 'data') 
else:
    raise NotImplementedError

assert os.path.exists(DIR_PROJECT) and os.path.exists(DIR_DATA)

############################################################
### general training options ###############################

MODEL = 'mlp'
N_FEATURES = 256 # number of features per network layer
N_LAYERS = 4 # number of hidden layers in network
N_INP = 3 # number of network inputs, i.e. 3D coordinates
N_OUT = 1 # number of network outputs, i.e. grayscale value at that coordinate

N_EPOCHS_MID = 2001 # n_epochs for reconstructing w randomly-init weights
N_EPOCHS_ALL = 401 #  ----------------------------- adjacent-init -------

LOSS_FN = 'l1' # loss function

GPU = 0
SEED = 0 # random seed for reproducibility

USE_LR_SCHEDULER = True
USE_HALF_PRECISION = True
USE_BATCHES = True
EPOCHS_TIL_CKPT = 200 # frequency of model checkpoints
EPOCHS_TIL_SUMMARY = 100 # frequency of tensorboard updates

N_SLICES_PER_NET = 2 # number of y-slices in a chunk
BATCH_SIZE = 2

CONFIGS_DATA_TYPE = {
    'sim': { # simulated datasets
        'lr': 1e-3,
        'lam_tv': 0,
        'n_sparse_prj': 61,
        'fn_tlt': 'tilt_angles.tlt',
    },
    'acq': { # experimentally-acquired datasets
        'lr': 1e-3,
        'lam_tv': 1., # total variation regularization
        'fn_tgt': '', # target volume n/a
    },
}

FN_METRICS = 'metrics.json' # file for image metrics
FN_OUT = 'out_val.pt' # reconstructed file


############################################################
### dataset-specific params ################################

DATASETS = ['sph', 'geo', 'p2s']
DATASETS_SIM = ['sph', 'geo', 'p2s'] # simulated datasets
DATASETS_ACQ = [] # datasets acquired experimentally (none provided by default)

CONFIGS_DATASET = {
    'sph': { # spheres
        'img_size_load': (1024, 1024, 256),
        'prj_size_load': (1024, 1024),
        'fn_prj': 'sph_prj.hdf',
        'fn_tgt': 'sph_tgt.hdf',
    },
    'geo': { # geometric shapes
        'img_size_load': (1024, 1024, 256),
        'prj_size_load': (1024, 1024),
        'fn_prj': 'geo_prj.hdf',
        'fn_tgt': 'geo_tgt.mrc',
    },
    'p2s': { # p22 particle
        'img_size_load': (360, 360, 360),
        'prj_size_load': (360, 360),
        'fn_prj': 'p2s_prj.mrc',
        'fn_tgt': 'p2s_tgt.mrc',

        'voxel_size': (360, 360, 360),
        'header_cella': (1620, 1620, 1620),
    },
}


############################################################
### experiment-specific configs ############################

cases = {
    99: { # throwaway case for making sure code didnt break
        #'n_epochs': 10,
    },

    0: {}, # default params

    # throwaway examples varying parameters from default
    1: {'seed': 1}, # random seed
    2: {            # number of epochs, learning rate
        'n_epochs': 1000,
        'lr': 5e-4,
    },

    ######################################
    ### p2s ablation expmts ##############

    # p2s with varied angular range, i.e. varying n_prjs
    11: { # n_prjs = 56
        'n_sparse_prj': 57,
        'fn_prj': 'ablation_range/prj_range56.pt',
        'fn_tlt': 'ablation_range/angles_range56.tlt',
    },
    12: { # n_prjs = 52
        'n_sparse_prj': 53,
        'fn_prj': 'ablation_range/prj_range52.pt',
        'fn_tlt': 'ablation_range/angles_range52.tlt',
    },
    13: { # n_prjs = 48
        'n_sparse_prj': 49,
        'fn_prj': 'ablation_range/prj_range48.pt',
        'fn_tlt': 'ablation_range/angles_range48.tlt',
    },
    14: { # n_prjs = 44
        'n_sparse_prj': 45,
        'fn_prj': 'ablation_range/prj_range44.pt',
        'fn_tlt': 'ablation_range/angles_range44.tlt',
    },
    15: { # n_prjs = 40
        'n_sparse_prj': 41,
        'fn_prj': 'ablation_range/prj_range40.pt',
        'fn_tlt': 'ablation_range/angles_range40.tlt',
    },
    
    # p2s with varied angular resolution
    21: { # project every 1 degree
        'n_sparse_prj': 121,
        'fn_prj': 'ablation_steps/prj_step1.hdf',
        'fn_tlt': 'ablation_steps/angles_step1.tlt',
    },
    22: { # project every 4 degrees
        'n_sparse_prj': 31,
        'fn_prj': 'ablation_steps/prj_step4.hdf',
        'fn_tlt': 'ablation_steps/angles_step4.tlt',
    },
    23: { # project every 8 degres
        'n_sparse_prj': 16,
        'fn_prj': 'ablation_steps/prj_step8.hdf',
        'fn_tlt': 'ablation_steps/angles_step8.tlt',
    },

}
