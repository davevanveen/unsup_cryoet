import argparse
import numpy as np
import os
import sys

import constants
import utils

def get_parser(purpose=None):
    ''' parse arguments '''

    p = argparse.ArgumentParser()

    ########## required command line inputs ##########
    p.add_argument('--dataset', type=str, required=True,
                   choices=['sph', 'geo', 'p2s'], help='cryoet dataset')
    p.add_argument('--expmt_type', type=str, required=True,
                   choices=['mid', 'all'],
                   help='construct a middle chunk only or all chunks in volume')
    p.add_argument('--case_id', type=int, default=0,
                   help='case_id for overwriting default args (optional)')

    ########## general training ##########
    p.add_argument('--path_init_model', type=str, default=None,
                   help='path to pretrained model')
    p.add_argument('--fit_img', type=str, default=None,
                   help='train directly on image located at this filename')
    p.add_argument('--idx_y', type=int, default=None,
                   help='''idx of middle y-slice to fit. default: middle slice
                           manually set to 0 if want entire volume''')
    p.add_argument('--n_epochs', type=int, default=None,
                   help='number of training epochs')
    p.add_argument('--seed', type=int, default=constants.SEED,
                   help='random seed for experiment reproducibility')
    p.add_argument('--gpu', type=int, default=constants.GPU, 
                   help='gpu id to use for training')
    p.add_argument('--lam_tv', type=float, default=0.,
                   help='strength of total variation loss term')

    ########## summary, logging ##########
    p.add_argument('--bypass_tb', action='store_true', default=False,
                   help='store output in separate folder to avoid tb load')

    ########## cryo-specific ##########
    p.add_argument('--mask_crow_divisor', type=int, default=None,
                   help='divisor for setting size of crowther mask')

    args = p.parse_args()

    args = set_args(args)

    if False: # print args to command line
        print('--- Run Configuration ---')
        for k, v in vars(args).items():
            print(k, v)

    return args


def set_args(args, purpose=None):
    ''' set args based on parser, constants.py '''
    
    if purpose == 'load': # irrelevant args necessary for loading
        args = set_irrelevant_args(args)
    args = transport_args(args) # dataset-specific from constants.py

    # set learning rate, n_epochs
    # match previous learning rate if resuming from initialized network
    if args.expmt_type == 'all':
        args.lr = args.lr / 10
    if not args.n_epochs: # set n_epochs if not in command line
        if args.expmt_type == 'all':
            args.n_epochs = constants.N_EPOCHS_ALL
        else:
            args.n_epochs = constants.N_EPOCHS_MID
    args.lrn = args.lr / 10 # learning rate at nth iteration
    args.lr_decay_start = args.n_epochs // 2
    args.lr_decay_end = args.n_epochs

    # overwrite default args w those in constants.cases
    args_dict = vars(args) # convert Namespace object --> dict
    for key, value in constants.cases[args.case_id].items():
        args_dict[key] = value
    args = argparse.Namespace(**args_dict) # convert dict --> Namespace object

    # determine input data directory and filenames
    args.dir_inp = os.path.join(constants.DIR_DATA, args.dataset)
    args.fn_prj = os.path.join(args.dir_inp, args.fn_prj)
    if args.fn_tgt:
        args.fn_tgt = os.path.join(args.dir_inp, args.fn_tgt)
    args.fn_tlt = os.path.join(args.dir_inp, args.fn_tlt)

    # n_miss_prj: number of missing projs on each side of wedge
    #             e.g. if 15, have projs for angles [15*2, 180-(15*2)]
    # n_sparse_prj: number of angle values in .tlt file
    # idx_prj: upper index for acquired projections
    args.n_prj = 90
    args.n_miss_prj = int(np.ceil((90 - args.n_sparse_prj) / 2))
    args.idx_prj = args.n_prj - args.n_miss_prj + 1

    # define size of img, projs after cropping desired y dimns
    if args.n_slices_per_net and args.fn_prj:
        args.img_size = (args.img_size_load[0], args.n_slices_per_net, args.img_size_load[2])
        args.prj_size = (args.prj_size_load[0], args.n_slices_per_net)
    else:
        args.img_size, args.prj_size = args.img_size_load, args.prj_size_load
    
    # determine output directory (including tb logs)
    suffix = '_bypass_tb' if args.bypass_tb else ''
    #args.dir_out_dataset = os.path.join(constants.DIR_PROJECT + suffix,
    #                                    args.dataset)
    args.dir_out_dataset = os.path.join(constants.DIR_PROJECT,
                                        'out' + suffix,
                                        args.dataset)
    args.dir_out = os.path.join(args.dir_out_dataset, f'c{args.case_id}')

    # if recon, set output directory as a particular slice
    if purpose != 'load':
        if args.idx_y is None: # set recon slice as middle if not specified
            args.idx_y = args.img_size_load[1] // 2
        idx_sy = str(args.idx_y).zfill(4)
        args.dir_out = os.path.join(args.dir_out, f'sy{idx_sy}')
        utils.cond_mkdir(args.dir_out)
        args.fn_out = os.path.join(args.dir_out, constants.FN_OUT)
        if os.path.exists(args.fn_out):
            print(f'already reconstructed {args.fn_out}')
            sys.exit()
    
    # determine steps_per_epoch based on batch_size, number of slices in y-axis
    if args.img_size[1] % args.batch_size != 0: # want to hit all points each iteration
        raise NotImplementedError('requires img_size_y is divisible by batch_size, \
                                   but got {}, {}, respectively'.format(args.img_size[1],
                                   args.batch_size))
    args.steps_per_epoch = args.img_size[1] // args.batch_size
    if purpose != 'load':
        msg = f'{args.img_size[1]} y-slices w batch_size {args.batch_size}'
        print(f'{msg} --> {args.steps_per_epoch} steps per epoch')

    if args.batch_size <= args.steps_per_epoch:
        print('training.py loop runs out of index')
    assert args.batch_size <= args.n_slices_per_net
    
    # transport arg from constants (default case)
    if 'n_features' not in vars(args):
        args.n_features = constants.N_FEATURES
    
    return args


def transport_args(args):
    ''' transport args from constants for convenience '''

    # init configs w default for sim/acq, then update for dataset-specific
    type_ = 'sim' if args.dataset in constants.DATASETS_SIM else 'acq'
    configs = constants.CONFIGS_DATA_TYPE[type_]
    configs.update(constants.CONFIGS_DATASET[args.dataset])

    args.img_size_load = configs['img_size_load']
    args.prj_size_load = configs['prj_size_load']
    args.n_sparse_prj = configs['n_sparse_prj']
    if 'lr' not in vars(args):
        args.lr = configs['lr']
    if 'lam_tv' not in vars(args):
        args.lam_tv = configs['lam_tv']

    # filenames
    args.fn_prj = configs['fn_prj']
    args.fn_tgt = configs['fn_tgt']
    args.fn_tlt = configs['fn_tlt']

    args.n_slices_per_net = constants.N_SLICES_PER_NET
    args.batch_size = constants.BATCH_SIZE

    return args

def set_irrelevant_args(args):
    ''' set args irrelevant for CryoLoad() '''

    args.bypass_tb = False
    args.n_epochs = None
    args.idx_y = 0
    args.expmt_type = 'mid'

    return args
