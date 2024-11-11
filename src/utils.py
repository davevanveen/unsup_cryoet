''' utility functions organized into two sections:
        1) write tensorboard summary
        2) plot and analyze output '''

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import os
import numpy as np
from torchvision.utils import make_grid
import torch.nn

import constants
import dataio
import metrics
import proc_freq


###############################################################################
### SECTION 1: write tensorboard summary
###############################################################################

def to_numpy(x):
    return x.detach().cpu().numpy()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_max_summary(name, tensor, writer, total_steps, mean_std=False):
    ''' write (min,max) value of tensor to tensorboard summary
        mean_std (boo): include (mean,std) of tensor '''

    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)
    if mean_std:
        tensor = tensor.float()
        writer.add_scalar(name + '_mean', tensor.mean().detach().cpu().numpy(), total_steps)
        writer.add_scalar(name + '_std', tensor.std().detach().cpu().numpy(), total_steps)


def write_metrics(img_out, img_tgt, writer, iter, prefix):

    img_out = img_out.cpu().numpy()
    img_tgt = img_tgt.detach().cpu().numpy()

    try: 
        scores = metrics.calc_metrics(img_tgt, img_out)
        psnr_ = scores['psnr_3d']
        ssim_ = scores['ssim']
        vif_ = scores['vif']
    except ZeroDivisionError:
        psnr_, ssim_, vif_ = 0, 0, 0

    val_scores = np.array((psnr_, ssim_, vif_))
    writer.add_scalar(prefix + 'psnr', psnr_, iter)
    writer.add_scalar(prefix + 'ssim', ssim_, iter)
    writer.add_scalar(prefix + 'vif', vif_, iter)

    return val_scores


def scale_to_0_1(t):
    ''' given torch tensor, scale to [0,1] for tensorboard display '''
    return (t - t.min()) / (t.max() - t.min())


def vol_to_slice(vol, plane='xy', idx=None, qdess=False):
    ''' given a 3d volume, return a 2d slice i
        if qdess, adjust aspect ratio (kx by 5, kz by 2 - approx) '''
    
    assert vol.ndim == 4 # [batch_size, kx, ky, kz]
    
    if not idx: # compute central idx for that plane
        plane_ = 1 if plane=='yz' else 2 if plane=='xz' else 3 # if plane == 'xy'
        idx = vol.shape[plane_] // 2

    if plane == 'yz':
        slice_ = vol[:, idx, :, :]
    elif plane == 'xz':
        slice_ = vol[:, :, idx, :]
    elif plane == 'xy':
        slice_ = vol[:, :, :, idx]
    else:
        raise NotImplementedError

    slice_ = scale_to_0_1(slice_)

    if qdess and plane != 'xy': # approx aspect ratio for in- v. thru-planes
        slice_ = scale_qdess_aspect_ratio(slice_)

    return slice_
    

def write_cryo_summary(image_resolution, fbp_recon, model, in_,
                       gt, writer, total_steps, args, is_sim=True,
                       eval_all_coords=False, val_dataloader=None):
    ''' tensorboard summary for cryo volume 
        args:
            in_: model input dict w coords 
            eval_all:   true: eval over entire coordinate grid 
                        false: eval over subset (memory efficient) '''

    assert val_dataloader is None # no val set for cryo, else return val_metrics 

    assert len(image_resolution) == 3
    image_resolution = (1,) + image_resolution # add batch dimn
        
    key_val = 'all' if eval_all_coords else 'val'

    with torch.no_grad():
      
        # load entire gt volume (only relevant if is_sim)
        vol_gt_all = gt['img_all'][[0]].squeeze(-1) # get first batch dimn, squeeze
         
        # load gt evaluation volume 
        if key_val == 'all': # evaluate over entire gt
            vol_gt = vol_gt_all
        else: # evaluate over subset of gt
            vol_gt = gt['img_{}'.format(key_val)][[0]].squeeze(-1)

        # compute model output over validation coordinates
        coords_val = in_['coords_{}'.format(key_val)] 
        if not coords_val.shape[0] == 1: # if multiple batches
            assert (torch.abs(coords_val[0] - coords_val[1]).max() == 0) 
        with torch.cuda.amp.autocast(enabled=constants.USE_HALF_PRECISION):
            out_ = model(coords_val[0]) # first dimn w dups per num_batches
        vol_pred = out_['model_out']['output'].squeeze().unsqueeze(0)

        if not is_sim: # don't have access to ground-truth for acquired data
            vol_gt = torch.empty(vol_pred.size())
        
        write_metrics(vol_pred, vol_gt, writer, total_steps, 'trn_img_')
       
        min_max_summary('trn_coords', coords_val, writer, total_steps)
        min_max_summary('trn_img_tgt', vol_gt, writer, total_steps, mean_std=True)
        min_max_summary('trn_img_out', vol_pred, writer, total_steps, mean_std=True)

        for plane in ['yz', 'xz', 'xy']: # for central plane in each direction
           
            # get coords for relevant plane
            coords_key = 'coords_{}'.format(plane)
            coords_slice = in_[coords_key] # may not be length 1 in batch dimn
            if not coords_slice.shape[0] == 1: # if multiple batches
                assert (torch.abs(coords_slice[0] - coords_slice[1]).max() == 0)
            coords_slice = coords_slice[0] # first dimn w duplicates per num_batches
            
            # compute model output for plane coordinates
            with torch.cuda.amp.autocast(enabled=constants.USE_HALF_PRECISION):
                out_slice = model(coords_slice)['model_out']['output']
            # [plane1, plane2, 1] --> [1, plane1, plane2]
            out_slice = scale_to_0_1(out_slice.squeeze().unsqueeze(0))

            # isolate relevant slice in gt, fbp for comparison. stack for tb output
            # if we don't have ground-truth, just add an empty tensor
            if is_sim:
                gt_slice = vol_to_slice(vol_gt_all, plane) # does 0_1 scaling
            else:
                gt_slice = torch.empty(out_slice.size()).to(out_slice.device)

            # create stack of slices to display in tensorboard
            if fbp_recon != None and args.dataset in constants.DATASETS_SIM:
                fbp_slice = vol_to_slice(fbp_recon, plane)
                slice_list = [gt_slice, fbp_slice.cuda(), out_slice]
            else:
                slice_list = [gt_slice, out_slice]
            output_vs_gt = torch.cat(slice_list, dim=-1)

            writer.add_image('trn_gt_vs_pred_{}'.format(plane),
                             # first line adjusts scale (default pre-220922), second doesn't
                             #make_grid(output_vs_gt, scale_each=False, normalize=True),
                             make_grid(output_vs_gt, scale_each=True, normalize=False), 
                             global_step=total_steps)

            if plane == 'xz': # plot fft images in missing wedge direction
                
                img_list = [dataio.scale_to_0_1(ii).float() for ii in slice_list]
                fft_list = [abs(proc_freq.fft_nd(ii, ndims=2)) for ii in img_list]
                
                # NOTE: b/c fft_out is changing, max val changing, so can't compare plot across iter
                # to fix, save updated max fft magnitude value at each eval. use this to scale
                c_fft = 200 # scale fft magnitude by constant so we can visualize in plot
                fft_list = [c_fft * dataio.scale_to_0_1(ii) for ii in fft_list]

                fft_list = torch.cat(fft_list, dim=-1)
                writer.add_image('fft_{}'.format(plane),
                                 make_grid(fft_list, scale_each=False, normalize=False),
                                 global_step=total_steps)


###############################################################################
### SECTION 2: plot and analyze output
###############################################################################

def make_colorbar_symmetric(arr):
    ''' modify array s.t. arr.max() = abs(arr.min())
        --> diverging colormap will render positives/negatives symmetrically
        
        find max absolute value in array 
        if positive, set [0,0] to largest negative number
        if negative, -------------------- positive number '''
    
    arr = np.array(arr)
    
    if abs(arr.max()) < abs(arr.min()):
        val = abs(arr.min())
    elif abs(arr.max()) >= abs(arr.min()):
        val = -abs(arr.max())
    else:
        raise ValueError
    
    arr[0,0] = val
    
    return arr


def fix_dynamic_range(diff_list, dyn_range=None):
    ''' fix dynamic range of diff map
        
        if   map has values greater than clip_val, clip to range 
        elif largest value of map is less than clip_val,
             set [-1,-1] pixel to clip_value '''
    
    if dyn_range == None:
        return diff_list
    
    for idx, diff in enumerate(diff_list):
        if diff.max() > dyn_range:
            diff_list[idx] = np.clip(np.array(diff), a_min=-dyn_range, a_max=dyn_range)
        else:
            diff[-1,-1] = dyn_range
            diff_list[idx] = diff
        
    return diff_list


def zero_low_freqs(diff_list):
    ''' zero out low freqs to more precisely analyze high freqs '''
    for idx, diff in enumerate(diff_list):
        assert (diff.size(0), diff.size(1)) == (160, 160)
        diff[40:120, 40:120] = 0
        diff_list[idx] = diff
    return diff_list


def plot_hist_stairs(a, plot_title='', bins=100, rm_first=False):
    ''' plot histogram of array '''
    counts, bins = np.histogram(a, bins=bins)
    if rm_first: # rm first count, bin
        counts, bins = counts[1:], bins[1:]
    plt.stairs(counts, bins, fill=True)
    plt.title(plot_title)
    plt.show()


def plot_row(arr_list, title_list=None, bbox_list=None,
             figsize=20, hist=False, clim=(0,1),
             reverse_cmap=False):
    ''' given list of imgs, plot a single row for comparison
    e.g. arr_list=[im_gt, im_1, im_2]
         title_list=[gt, method 1, method 2] '''

    num_cols = len(arr_list)
    if not title_list:
        title_list = num_cols * ['']
    if bbox_list is None:
        bbox_list = num_cols * [False]

    fig = plt.figure(figsize=(figsize,figsize))

    cmap = 'gray_r' if reverse_cmap else 'gray'

    for idx_s in range(num_cols):
        ax = fig.add_subplot(1,num_cols,idx_s+1)
        if hist:
            n, bins, patches = ax.hist(arr_list[idx_s], bins=25, density=True,
                                       facecolor='g', alpha=0.75)
        else:
            ax.imshow(arr_list[idx_s], cmap=cmap, clim=clim)
        ax.set_title(title_list[idx_s], fontsize=20)
        ax.axis('off')


def plot_row_(arr_list, title_list=None, inset_list=None,
             bbox_list=None, figsize=12, clim=None, clim_ins=None,
             cmap_gray=True, is_diff_map=False,
             vmin=None, vmax=None):
    ''' note: similar function w options for midl paper plots, 202204

        given list of imgs, plot a single row for comparison
        e.g. arr_list=[im_gt, im_1, im_2]
             title_list=[gt, method 1, method 2] 
         
        inset_list: list of inset images w clim_ins '''
    
    num_cols = len(arr_list)
    if not title_list:
        title_list = num_cols * ['']
    if bbox_list is None:
        bbox_list = num_cols * [False]
        
    fig = plt.figure(figsize=(figsize,figsize))

    for idx_s in range(num_cols):
        ax = fig.add_subplot(1,num_cols,idx_s+1)
        if cmap_gray:
            ax.imshow(arr_list[idx_s], cmap='gray', clim=clim,
                                       vmin=vmin, vmax=vmax)
        else: # color plot w colorbar
            if is_diff_map: # make colorbar symmetric
                arr_list[idx_s] = make_colorbar_symmetric(arr_list[idx_s])
            plot_ = ax.imshow(arr_list[idx_s], cmap='bwr')
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(plot_, cax=cax)
        ax.set_title(title_list[idx_s], fontsize=20)
        ax.axis('off')
        
        if bbox_list[idx_s]:
            y0, x0, size = 216, 216, 80 # 236, 236, 40 = central [236,276]
            rect = patches.Rectangle((y0,x0),size,int(size),
                                     linewidth=3,edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            
        if inset_list != None: 
            
            inset = inset_list[idx_s]
            if inset == None:
                continue
            
            # manually set axes locations depending on num_plots in row
            # see https://stackoverflow.com/questions/43326680/what-are-the-differences-between-add-axes-and-add-subplot
            if num_cols == 1:
                ax_loc_list = [[0.675,0.675,0.175,0.175]]
            if num_cols == 2:
                size_ = .075
                height_ = 0.5875
                ax_loc_list = [[0.3875,height_,size_,size_],
                               [0.8125,height_,size_,size_]]
            if num_cols == 3:
                size_ = .075
                height_ = 0.535
                ax_loc_list = [[0.2705,height_,size_,size_],
                               [0.545,height_,size_,size_],
                               [0.81875,height_,size_,size_]]
            if num_cols > 3:
                raise NotImplementedError
            
            inset = inset_list[idx_s]
            ax_loc = ax_loc_list[idx_s]
            
            newax = fig.add_axes(ax_loc, anchor='NE')
            newax.imshow(inset, cmap='gray', clim=clim_ins)
            plt.rcParams["axes.edgecolor"] = 'green'
            plt.rcParams["axes.linewidth"]  = 1
            newax.tick_params(axis='x', which='both', bottom=False, top=False)
            newax.tick_params(axis='y', which='both', left=False, right=False)
            plt.xticks(visible=False), plt.yticks(visible=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fft_plot(a, fft_scale=50):
    return fft_scale*dataio.scale_to_0_1(abs(proc_freq.fft_nd(a, ndims=2)))


def plot_slice_xyz(arr_list,
                   num_slices_total=[10,10,10], num_slices_per_row=[5,10,10],
                   figsize_list=[32,32,2], plot_fft=False, fft_scale=50):
    ''' given 3d array, plot central slice in each dimn
        num_slices_total: how many slices to plot in each dimn '''
    
    len_xyz = arr_list[0].shape

    # for each axis, get indices to display
    idcs_xyz = []
    for idx_, len_ in enumerate(len_xyz):

        # get boundaries for indices
        idx_0, idx_1 = int(.25 * len_), int(.75 * len_)

        if num_slices_total[idx_] == None: # if not specified, plot every index
            num_slices_total[idx_] = idx_1 - idx_0

        # get all indices
        ii = np.linspace(idx_0, idx_1, num_slices_total[idx_]).astype('int')

        # first crop last few indcs s.t. len(idcs) % 10 == 0
        ii = ii[0 : len(ii) - (len(ii) % num_slices_per_row[idx_])]
        # then reshape so we get appropriate number of slices per row
        ii = ii.reshape(-1, num_slices_per_row[idx_])

        idcs_xyz.append(ii)

    xx, yy, zz = idcs_xyz
    fs_x, fs_y, fs_z = figsize_list

    # plot individual planes
    for xx_row in xx:
        [plot_row([arr[x_, :, :] for x_ in xx_row], figsize=fs_x) for arr in arr_list]
        if plot_fft:
            [plot_row([fft_plot(arr[x_, :, :], fft_scale) for x_ in xx_row], figsize=fs_x) for arr in arr_list]
    for yy_row in yy:
        [plot_row([arr[:, y_, :] for y_ in yy_row], figsize=fs_y) for arr in arr_list]
        if plot_fft:
            [plot_row([fft_plot(arr[:, y_, :], fft_scale) for y_ in yy_row], figsize=fs_y) for arr in arr_list]
    for zz_row in zz:
        [plot_row([arr[:, :, z_] for z_ in zz_row], figsize=fs_z) for arr in arr_list]
        if plot_fft:
            [plot_row([fft_plot(arr[:, :, z_], fft_scale) for z_ in zz_row], figsize=fs_z) for arr in arr_list]
