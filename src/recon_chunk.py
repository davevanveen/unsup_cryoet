from functools import partial
import numpy as np
import os
import sys
import time
import torch

import constants
import cryo_geometry
import dataio
import dataset
import loss_functions
import modules
import parser
import training
import utils


def main():

    # set args, cuda configs
    args = define_preliminaries()

    # define training data
    data = dataset.CryoTrain(args)

    # define loss and tb summary functions
    loss_fn = loss_functions.get_loss_fn(constants.LOSS_FN)
    summary_fn = partial(utils.write_cryo_summary, 
                         args.img_size, data.fbp)

    # define model
    model = define_model(args)
    save_expmt_params(args, model)
   
    # train model weights with unsupervised method
    t0 = time.time()
    training.train(args, model, data, loss_fn, summary_fn) 
    print(f'training time {time.time()-t0}')

    # query all coordinated of trained model 
    eval_model(args, model, coords=data.coords)


def eval_model(args, model, coords):
    ''' evaluate model at given coordinates 
        args:   model: untrained model '''

    # load model
    eval_epoch = args.n_epochs - 1
    model_path = os.path.join(args.dir_out,
                    #'checkpoints/model_epoch_%.04d.pth' % eval_epoch)
                    'checkpoints/model_final.pth')
    model.load_state_dict(torch.load(model_path))

    # evaluate at given coords
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=constants.USE_HALF_PRECISION):
        out = model({'coords': coords.cuda()})['model_out']['output'] 
    out = out.reshape(args.img_size)
    if args.dataset != 'vac':
        out = torch.clip(out, 0, 1)
    
    torch.save(out.detach().cpu(), args.fn_out)
    print(f'completed expmt in {args.dir_out}')


def define_preliminaries():
    ''' define configs, set cuda device and seed value '''
    
    args = parser.get_parser()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    # set cuda device, random seed
    torch.cuda.set_device(args.gpu)
    if args.seed != None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    return args


def define_model(args):

    assert constants.MODEL == 'mlp'
        
    m = modules.CoordinateNet 
    model = m(constants.N_INP, args.n_features, constants.N_OUT,
              num_hidden_layers=constants.N_LAYERS,
              nl='sine', pe_scale=4, no_pe=False)
    #print('model has {} params'.format(utils.count_parameters(model)))
    model.cuda()

    if args.path_init_model != None:
        model.load_state_dict(torch.load(args.path_init_model))
        print('loaded pretrained model {}'.format(args.path_init_model))

    return model


def save_expmt_params(args, model):

    # save command-line parameters log directory
    fn_params = os.path.join(args.dir_out, 'params.txt')
    with open(fn_params, 'w') as f:
        for arg_name in vars(args):
            f.write(f'{arg_name} {getattr(args, arg_name)}\n')

    # save text summary of model into log directory
    with open(os.path.join(args.dir_out, 'model.txt'), 'w') as out_file:
        out_file.write(str(model))


def generate_prjs(args):
    ''' if prjs not pre-defined, generate re-prjs w img + ct_projector
        note: this auxiliary function is currently not called '''

    img = dataio.load_img_file(args, args.fn_fbp)
    img = img.squeeze().unsqueeze(0) # [1,x,y,z]

    ct_projector = cryo_geometry.Parallel3DProjector(args)
    prjs = ct_projector.forward_project(img, args).squeeze()
    prjs = dataio.scale_to_0_1(prjs)
    
    fn_out = os.path.join(args.dir_inp, args.fn_reprjs)
    torch.save(prjs, fn_out) 
    print(f'generated prjs. re-run script w {fn_out} as fn_prj')

    # optional: back-project into image space
    #fbp = ct_projector.backward_project(prjs.unsqueeze(0))

    sys.exit()


if __name__ == '__main__':
    main()
