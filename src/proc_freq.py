import numpy as np
import torch
import bm3d

import dataio


def run_bm3d(im, sigma=0.1, is_training=True):
    ''' stages_arg: perform hard_thresholding (.92s) 
                            or all_stages (1.92s, slightly better quality) 
                    time evaluated on 512x512 grayscale image 
        is_training: true if function being called from training loop'''
   
    if sigma == 0:
        return im

    if is_training: # less expensive operation if in training loop
        stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
    else:
        stage_arg=bm3d.BM3DStages.ALL_STAGES
    
    return bm3d.bm3d(im, sigma_psd=sigma, stage_arg=stage_arg) 


def add_noise_in_ksp(arr, noise_std=0.1):
    ''' given image, add gaussian noise in fourier space
        note: if undersampled mri scan, dont add noise to masked region '''

    ksp = fft_nd(arr, ndims=2)

    if 'complex' in str(ksp.dtype):
        noise = noise_std * torch.randn(ksp.shape + (2,))
        noise = torch.view_as_complex(noise)
    else:
        noise = noise_std * torch.randn(ksp.shape)

    ksp_n = ksp + noise

    arr_n = abs(ifft_nd(ksp_n, ndims=2))

    return arr_n


def sinc_interpolate(arr, factor):
    ''' given real-valued image arr 
        zero pad in freq domain <--> interpolate in im domain 
        return real-valued image upsampled by factor '''
    
    assert arr.ndim == 2
    assert arr.shape[0] == arr.shape[1]
    arr = torch.tensor(arr)
    
    freq = fft_nd(arr, ndims=2)

    # num zeros to pad on each side
    len_curr = freq.shape[0]
    len_end = int(factor * len_curr)
    len_pad = (len_end - len_curr) // 2
    pad_2d = (4*(len_pad, )) # assume square pad
    
    freq_zf = torch.nn.functional.pad(freq, pad_2d, mode='constant')

    return abs(ifft_nd(freq_zf, ndims=2))


def apply_consistency(vol, dim='y'):
    ''' given a 3d volume
        apply a gaussian kernel in specified dimn
        e.g. can apply this to achieve slice-wise consistency in y'''

    # permute tuple so we convolve over last dimension
    perm = (0,1,2) if dim == 'z' else \
           (0,2,1) if dim == 'y' else \
           (2,1,0) # if dim == 'x'

    # convolution w a 1x1x5 kernel will perform convolution over third dimn
    mod = torch.nn.Conv1d(1, 1, 5, padding=2, bias=False, groups=1)

    ker = torch.tensor([.02, 0.36788, 1, 0.36788, .02])[None, None, :]
    mod.weight.data = torch.nn.functional.normalize(ker)

    vol = torch.permute(vol, perm) # so we convolve over last dimn (y)
    out_vol = torch.empty_like(vol)

    # can't do conv1d w 3d input, so proceed by slice
    for idx_x, vol_x in enumerate(vol):
        out_vol[idx_x] = mod(vol_x[:, None, :]).squeeze().detach()

    out_vol = torch.permute(out_vol, perm) # convert back to x,y,z order

    # # alternatively, via scipy
    # out_ker_ = scipy.ndimage.convolve1d(out_vol, ker, axis=1, mode='nearest')
    # out_ker_ = torch.tensor(out_ker_)

    return out_vol


def filter_image(arr, perc=.5, filter_type='lo'):
    ''' given 2d, square image, apply low- or high-pass filter 
        i.e. convert to freq space, get central crop indices, then ...
             lo: only pass central square
             hi: zero out central square '''
    
    assert arr.ndim == 2
    assert arr.shape[0] == arr.shape[1]
    
    # determine indices to crop in k-space
    orig_size = arr.shape[0]
    crop_size = int(perc * orig_size)
    ind_x0, ind_x1 = dataio.crop_indices(crop_size, orig_size)

    arr = torch.tensor(arr)
    freq = fft_nd(arr, ndims=2)    
    
    if filter_type == 'lo': # low-pass filter
        freq_lo = torch.zeros((freq.shape)).type(torch.complex64)
        freq_lo[ind_x0:ind_x1, ind_x0:ind_x1] = freq[ind_x0:ind_x1, ind_x0:ind_x1]
        freq = freq_lo
    
    elif filter_type == 'hi': # high-pass filter
        # e.g. perc = .5 --> 0:128, 384:512 non-zero
        freq[ind_x0:ind_x1, ind_x0:ind_x1] = 0
    
    return abs(ifft_nd(freq, ndims=2)), abs(freq)


def ifft_nd(arr, ndims):
    ''' apply centered fft over the last ndims dimensions of arr '''

    assert is_complex(arr) # input must be complex-valued

    dims = tuple(np.arange(-ndims, 0))

    arr = ifftshift(arr, dim=dims)
    # note: added norm='ortho' for fastmri expmt, but didn't have that arg for qdess
    arr = torch.fft.ifftn(arr, dim=dims, norm='ortho')
    arr = fftshift(arr, dim=dims)

    return arr


def fft_nd(arr, ndims):
    ''' apply centered fft over the last ndims dimensions of arr '''

    # assert is_complex(arr) # input must be complex-valued
    if not torch.is_tensor(arr):
        arr = torch.tensor(arr)

    dims = tuple(np.arange(-ndims, 0))

    arr = ifftshift(arr, dim=dims)
    # note: added norm='ortho' for fastmri expmt, but didn't have that arg for qdess
    arr = torch.fft.fftn(arr, dim=dims, norm='ortho')
    arr = fftshift(arr, dim=dims)

    return arr


def root_sum_squares(arr):
    ''' given 3d complex arr [nc,x,y], perform rss over magnitudes
        return 2d arr [x,y] '''

    assert is_complex(arr)
    return torch.sqrt(torch.sum(torch.square(abs(arr)), axis=0))


def is_complex(arr):
    dt = arr.dtype
    return dt==torch.complex64 or dt==torch.complex128 or dt==torch.complex32


##########################################################
# helper functions internal to script, i.e. not for export


def ifftshift(x, dim=None):
    ''' Similar to np.fft.ifftshift but applies to PyTorch Tensors '''
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fftshift(x, dim=None):
    ''' similar to np.fft.fftshift but applies to PyTorch Tensors '''
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def roll(x, shift, dim):
    ''' similar to np.roll but applies to PyTorch Tensors '''
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)