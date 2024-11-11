import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset

import constants
import cryo_geometry
import dataio
import metrics
import utils


class CryoBase(Dataset):

    def __init__(self, args): 
        ''' instantiate CryoBase class for one of two sub-classes
            CryoTrain:  class for fitting dataset
            CryoLoad:   class for loading, analyzing expmt outputs '''

        self.img_size = args.img_size # according to crop_y dims

        # load projections
        self.prj = dataio.load_prj_file(args) # should be scaled to [0,1]

        self.is_sim = args.dataset in constants.DATASETS_SIM
        if self.is_sim:
            self.tgt = dataio.load_img_file(args, args.fn_tgt)
        else:
            self.tgt = torch.empty(1)


class CryoLoad(CryoBase):

    def __init__(self, args): 
        ''' class for loading expmt output, analyzing ''' 

        super().__init__(args) # i.e. CryoBase __init__()

        # load expmt output
        self.get_subdirs(args)
        self.out = self.load_recon(args)
        cond = self.len_y==self.out.shape[1]
        assert cond, f'array shape {self.out.shape} incompatible w len_y {self.len_y}'

        # ensure appropriate volume sizes
        self.tgt = self.tgt.squeeze()
        if self.is_sim: # crop tgt according to size of out
            self.tgt = self.crop_img(self.tgt.squeeze())

        # postprocess, save to mrc
        self.scale_to_0_1() # scale all volumes to be on [0,1]
        self.apply_rotate_and_flip(args) # align w jesus input
        self.save_to_mrc(args)


    def get_subdirs(self, args, filter_str='sy'):
        ''' return all immediate subdirectories of the given dir_
            filter (str): remove all subdirs not containing this str '''

        subdirs = [name for name in os.listdir(args.dir_out)
                if os.path.isdir(os.path.join(args.dir_out, name))]

        # only include subdirs w 'sy' in name and containing FN_OUT
        subdirs = [ss for ss in subdirs if filter_str in ss]
        subdirs = [ss for ss in subdirs if os.path.exists(
                                           os.path.join(args.dir_out, ss, constants.FN_OUT))]

        # filter out original middle slice recon w uninitialized net
        subdirs = [ss for ss in subdirs if '_no-init' not in ss]

        if len(subdirs) == 0:
            raise FileNotFoundError(f'no files in subdir {args.dir_out}')

        subdirs = [os.path.join(args.dir_out, ss) for ss in subdirs]
        subdirs.sort()
        self.subdirs = subdirs

        self.y0 = get_y_idx(subdirs[0])
        self.y1 = get_y_idx(subdirs[-1]) + args.n_slices_per_net
        self.len_y = self.y1 - self.y0


    def load_recon(self, args, load_fbp=False):
        ''' load indiv 3d chunks into a single 3d volume '''

        fn = '/fbp_recon.pt' if load_fbp else f'/{constants.FN_OUT}'
        arr_list = [torch.load(dd + fn).float() for dd in self.subdirs]

        # create empty tensor volume
        len_x, len_y, len_z = arr_list[0].shape
        vol = torch.empty((len_x, len_y * len(arr_list), len_z))

        # populate empty array w smaller 3d chunks
        for idx, arr in enumerate(arr_list):
            y0 = args.n_slices_per_net * idx
            y1 = args.n_slices_per_net * (idx + 1)
            vol[:, y0:y1, :] = arr

        return vol


    def apply_rotate_and_flip(self, args):
        ''' wrapper to rotate/flip each volume s.t. we align w input '''

        self.out = rotate_and_flip(self.out, args)
        if self.is_sim:
            self.tgt = rotate_and_flip(self.tgt, args)


    def save_to_mrc(self, args):
        ''' save outputs to mrc file '''

        dir_mrc = os.path.join(args.dir_out, 'mrc')
        utils.cond_mkdir(dir_mrc)

        fn_lst = ['out']
        fn_lst += ['tgt'] if self.is_sim else []
        fn_lst = [f'{dir_mrc}/{fn}.mrc' for fn in fn_lst]

        arr_lst = [self.out]
        arr_lst += [self.tgt] if self.is_sim else []

        for fn, arr in zip(fn_lst, arr_lst):
            dataio.save_arr(fn, arr,
                            shift_resize=True, dataset=args.dataset)


    def crop_img(self, img):
        ''' crop central y-slices according to recon size '''
        return img[:, self.y0:self.y1, :]


    def scale_to_0_1(self):
        ''' scale voxel values to the range of [0, 1] '''
        self.out = dataio.scale_to_0_1(self.out)
        if self.is_sim:
            self.tgt = dataio.scale_to_0_1(self.tgt)


    def calc_scores(self, args):
        ''' calculate or load metric scores '''

        fn_metrics = os.path.join(args.dir_out, constants.FN_METRICS)
        
        if not os.path.exists(fn_metrics): # calculate metrics if dne
            
            print(f'calculating metrics, {args.dir_out}')
            scores = {}
            scores['out'] = metrics.calc_metrics(self.tgt, self.out)

            with open(fn_metrics, 'w') as f:
                json.dump(scores, f)
            self.scores = scores

        else: # load pre-calculated metrics
            with open(fn_metrics, 'r') as f:
                self.scores = json.load(f)


class CryoTrain(CryoBase):

    def __init__(self, args): 
        ''' class for training i.e. running expmts ''' 
        
        super().__init__(args) # i.e. CryoBase __init__()

        self.batch_size = args.batch_size 
        self.img_size = args.img_size # according to crop_y dims

        # crop central y-slices (if args.n_slices_per_net != 0)
        self.y0, self.y1 = get_y_inds(args)
        self.prj = self.prj[:, :, self.y0:self.y1] # [n_sparse_prj, x, y]
        if self.is_sim:
            self.tgt = self.tgt[:, self.y0:self.y1, :, :] # [x, y, z, 1]

        self.fbp, self.ct_projector = back_proj(args, self.prj) # must crop first
        self.fbp = self.fbp.permute(3, 0, 1, 2) # TODO: move to appropriate place?
        self.coords = create_grid_3d(args) # coordinate grid
        self.set_val_attr() # attributes for validation during training


    def set_val_attr(self):
        ''' set attributes for validation during training '''
        
        # get central sub-volume for validation metrics to reduce memory
        mid_pts = [self.img_size[ii] // 2 for ii in [0,1,2]]
        mid_x, mid_y, mid_z = mid_pts
        # reduce each axis by factor ff, clipped on [c_min,c_max]
        ff, c_min, c_max = 4, 4, 64 # >=2*4 for metrics, <=2*64 for memory 
        len_x, len_y, len_z = [max(c_min, min(c_max, mm // ff)) for mm in mid_pts]
        self.coords_val = self.coords[mid_x-len_x:mid_x+len_x,
                                      mid_y-len_y:mid_y+len_y,
                                      mid_z-len_z:mid_z+len_z, :]
        if self.is_sim:
            self.tgt_val = self.tgt[mid_x-len_x:mid_x+len_x,
                                    mid_y-len_y:mid_y+len_y,
                                    mid_z-len_z:mid_z+len_z]
        else:
            self.tgt_val = torch.empty(1)
        
        # get central slice in each plane for validation
        self.coords_xy = self.coords[:, :, mid_z, :]
        self.coords_xz = self.coords[:, mid_y, :, :]
        self.coords_yz = self.coords[mid_x, :, :, :]


    def __len__(self):
        return self.img_size[1] // self.batch_size


    def __getitem__(self, idx):

        # get random y-slices per each training step
        idcs_y = np.random.choice(self.img_size[1], self.batch_size, replace=False)
        coords = self.coords[:, idcs_y, :, :]

        img = self.tgt[:, idcs_y, :, :] if self.is_sim else torch.empty(1)
        prj = self.prj[:, :, idcs_y]

        in_dict = {'coords': coords, 'coords_all': self.coords,
                   'coords_xy': self.coords_xy, 
                   'coords_xz': self.coords_xz,
                   'coords_yz': self.coords_yz,
                   'coords_val': self.coords_val}
        gt_dict = {'img': img, 'prj': prj, 
                   'img_all': self.tgt, 'img_val': self.tgt_val}

        return in_dict, gt_dict


def get_y_inds(args):
    ''' get y indices [y0,y1] for training volume '''

    if not args.n_slices_per_net:
        y0, y1 = 0, args.img_size_load[1]
    
    else:
        
        if not args.idx_y:
            idx_y = args.img_size_load[1] // 2
        else:
            idx_y = args.idx_y
        
        y0 = idx_y
        y1 = idx_y + args.n_slices_per_net
        
    return y0, y1
    

def get_y_idx(subdir):
    ''' given a subdirectory ending in syXXX, return XXX '''
    assert 'sy' in subdir
    return int(subdir.split('sy')[1])


def back_proj(args, prj):
    ''' perform back projection '''

    # instantiate projection operator, independent of img, prj
    ct_projector = cryo_geometry.Parallel3DProjector(args)

    prj = prj.unsqueeze(0) 
    fbp_recon = ct_projector.backward_project(prj.detach().clone())
    #torch.save(fbp_recon[0], os.path.join(args.dir_out, 'fbp_recon.pt'))

    # [1,x,y,z] --> [x,y,z,1] to match self.img.shape
    fbp_recon = dataio.scale_to_0_1(fbp_recon.squeeze(0).unsqueeze(-1))
    assert(len(fbp_recon.shape) == 4)

    return fbp_recon, ct_projector


def create_grid_3d(args, load_all=False, get_int_coords=False):
    ''' load_all: return coords across entire volume
                  else just coords of y-chunk '''

    c, h, w = args.img_size_load if load_all else args.img_size

    if get_int_coords: # side expmt to try re-indexing in training loop
        grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, c-1, steps=c), \
                                        torch.linspace(0, h-1, steps=h), \
                                        torch.linspace(0, w-1, steps=w)])
    else:
        grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                                torch.linspace(0, 1, steps=h), \
                                                torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    
    #torch.save(grid, os.path.join(args.dir_out, 'coords_trn.pt'))

    return grid


def rotate_and_flip(arr, args):
    ''' rotation + flip post-processing operations s.t. recons align w input '''

    if True: #args.dataset in constants.DATASETS_TO_ROTATE_FLIP:
        
        rotation = 1 # rotate +90 degrees
        rot_axes = (0, 2)
        flip_axis = 0 # flip about x-axis
        
        arr = torch.rot90(arr, k=rotation, dims=rot_axes).flip(flip_axis)
        
    return arr