''' data i/o and helper functions. organized into two sections:
        1) file input/output 
        2) miscellaneous helper functions '''

import h5py
import mrcfile
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import torch

import constants


###############################################################################
### SECTION 1: load inputs, basic data i/o
###############################################################################

def load_prj_file(args):
    ''' load projections i.e. tilt series '''

    if args.fn_prj is None:
        raise NotImplementedError('see run.generate_prjs()')
    
    file_type = args.fn_prj.split('.')[-1]

    if file_type == 'pt': # pre-processed projections
        if args.dataset == 'vac':
            # for vaccine, maybe we don't want to scale to [0,1]? tbd
            raise NotImplementedError('processing has changed')
        return load_arr(args.fn_prj, scale=True) # prjs must be on [0,1]

    elif file_type == 'hdf':
        prjs = []
        for i in range(args.n_sparse_prj):
            f = h5py.File(args.fn_prj, 'r')
            prj = f['MDF']['images'][str(i)]['image'][()]
            prjs.append(prj)
        prjs = np.stack(prjs, axis=0)
    
    elif file_type == 'mrc':
        mrc = mrcfile.open(args.fn_prj)
        prjs = torch.tensor(mrc.data)

    elif file_type == 'npz':
        prjs = np.load(args.fn_prj)['data']
        prjs = torch.tensor(prjs, dtype=torch.float32)
    
    elif file_type == 'npy':
        prjs = np.load(args.fn_prj)
        prjs = torch.tensor(prjs, dtype=torch.float32)
    
    else:
        raise NotImplementedError(f'file type {file_type} not supported')

    # some sim prjs have 90 angle-steps --> drop those outside central angle
    n_sim_prj = prjs.shape[0]

    if n_sim_prj != args.n_sparse_prj:
        assert n_sim_prj == 90
        prjs = prjs[args.n_miss_prj:args.idx_prj, :, :]

    # rotate volume to fit geometry. want shape [n_sparse_prj, x, y]
    prjs = torch.tensor(np.array(prjs)[::-1, ::-1, :].copy())
    prjs = torch.rot90(prjs, k=-1, dims=[1, 2]) 
    assert(prjs.shape == (args.n_sparse_prj,) + args.prj_size_load)

    return prjs


def load_img_file(args, fn):
    ''' load image file of 3d tomogram '''

    file_type = fn.split('.')[-1]

    if file_type == 'pt':
        arr = torch.load(fn) # generated .pt files already pre-processed
        return arr
    else:
        image = load_arr(fn, scale=False)

    # obtain desired geometry: [x,y,z,1]
    image = image.transpose(1, 3) 
    img = image.permute(1, 2, 3, 0)
    assert(img.shape == args.img_size_load + (1,)) # [x,y,z,1]

    return img


def load_arr(filename, return_as_type=None, scale=True):
    ''' given absolute path to array
        load, normalize, return as torch tensor 
                                or PIL img if return_as_type==img '''

    file_type = filename.split('.')[-1]

    if file_type == 'npy':
        arr = np.load(filename).astype('float32')
    elif file_type == 'pt':
        arr = torch.load(filename)
    elif file_type == 'png' or file_type == 'jpg':
        img = Image.open(filename).convert('L') # assumes grayscale
        arr = np.array(img).astype(np.float32)
    elif file_type == 'hdf':
        with h5py.File(filename, 'r') as f:
            arr = f['MDF']['images'][str(0)]['image'][()]
        arr = np.array(arr).astype(np.float32)
        arr = torch.tensor(arr, dtype=torch.float32)[None, ...]  # [B, C, H, W]
    elif file_type == 'mrc' or file_type == 'rec':
        arr = mrcfile.open(filename).data.astype(np.float32)
        arr = torch.tensor(arr, dtype=torch.float32)[None, ...] # [B, C, H, W]
    elif file_type == 'npz':
        arr = np.load(filename)['data']
        arr = torch.tensor(arr, dtype=torch.float32)[None, ...]  # [B, C, H, W]
    else:
        raise NotImplementedError
    
    if return_as_type == 'img': # return as PIL Image
        return convert_arr_to_img(arr)
    
    if scale:
        arr = scale_to_0_1(torch.tensor(arr))

    return arr


def save_arr(filename, arr,
             shift_resize=False, dataset=None):
    ''' given absolute path and array
        shift_resize: option to shift and resize volume for mrc file
        dataset: only necessary if rescaling
        save according to file extension '''

    file_ext = filename.split('.')[-1]

    if file_ext == 'npy':
        np.save(filename, arr)

    elif file_ext == 'pt':
        torch.save(arr, filename)

    elif file_ext == 'png' or file_ext == 'jpg':
        img = convert_arr_to_img(np.array(arr))
        img.save(filename) # assumes grayscale

    elif file_ext == 'mrc':
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(np.array(arr))

            config_ = constants.CONFIGS_DATASET[dataset]

            # shift and resize data so recon aligns w input
            cond_ = 'voxel_size' in config_ and 'header_cella' in config_
            if shift_resize and cond_:
                mrc.voxel_size = config_['voxel_size']
                mrc.header.cella.x = config_['header_cella'][0]
                mrc.header.cella.y = config_['header_cella'][1]
                mrc.header.cella.z = config_['header_cella'][2]

            if 'origin' in config_: 
                mrc.header.origin.x = config_['origin'][0]
                mrc.header.origin.y = config_['origin'][1]
                mrc.header.origin.z = config_['origin'][2]

            if 'nxyz_start' in config_:
                mrc.header.nxstart = config_['nxyz_start'][0]
                mrc.header.nystart = config_['nxyz_start'][1]
                mrc.header.nzstart = config_['nxyz_start'][2]

    else:
        raise NotImplementedError
    
    return


###############################################################################
### SECTION 2: miscellaneous helper functions
###############################################################################


def display_arr_stats(tensor):
    if type(tensor) is np.ndarray:
        tensor = torch.from_numpy(tensor)
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def print_tensor_size(tensor, prefix=''):
    ''' given tensor, print its memory footprint '''
    bytes_ = tensor.element_size() * tensor.nelement()
    mbytes = bytes_ / 1e6
    print('{}requires {}mb memory'.format(prefix, mbytes))


def scale_to_0_1(t):
    ''' scale tensor values to [0,1] '''
    return (t - t.min()) / (t.max() - t.min())


def convert_arr_to_img(arr):
    ''' given np array, convert to grayscale PIL Image '''
        
    arr = scale_to_0_1(np.array(arr)) # ensure range [0,1]
    img = Image.fromarray((255 * arr).astype(np.uint8)).convert('L')

    return img


def get_file_list(path, abs_path=False):
    ''' get list of files in a directory
        abs_path: if True, returns absolute path instead of relative path '''
    files = [ff for ff in listdir(path) if isfile(join(path, ff))]
    files.sort()
    if abs_path: # return absolute path
        files = [path + ff for ff in files]
    return files