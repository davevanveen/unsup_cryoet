''' functions to compute image quality metrics '''

import scipy
import numpy as np
import torch
import torch.nn as nn
import math, numbers
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#from pytorch_msssim import msssim


def calc_metrics(img_tgt, img_out, hfen=None, msssim=None):
    ''' compute vif, mssim, ssim, and psnr of img_out using
        img_tgt as ground-truth target reference '''

    img_tgt, img_out = np.float32(img_tgt), np.float32(img_out)
    img_tgt, img_out = norm_imgs(img_tgt, img_out)
    img_tgt, img_out = np.array(img_tgt), np.array(img_out)
    
    if min(img_tgt.shape) < 7: # need at least 7x7 for ssim
        ssim_lst = []
        for idx_y in range(img_tgt.shape[1]):
            y_tgt = img_tgt[:, idx_y, :]
            if min(y_tgt.shape) < 7: # if still not > 7x7 in 2d, skip
                ssim_lst.append(0)
            y_out = img_out[:, idx_y, :]
            ssim_lst.append(round(float(ssim(y_tgt, y_out)), 3))
        ssim_ = round(np.mean(np.array(ssim_lst)), 3)
    else:
        ssim_ = round(float(ssim(img_tgt, img_out)), 3)

    vif_ = round(float(vifp_mscale(img_tgt, img_out, sigma_nsq=img_out.mean())), 3)

    # compute psnr over each y-slice. if -inf (i.e. empty space), enter 0
    psnr_2d_lst = []
    for idx_y in range(img_tgt.shape[1]):
        y_tgt = img_tgt[:, idx_y, :]
        y_out = img_out[:, idx_y, :]
        psnr_2d = psnr(y_tgt, y_out)
        if not np.isinf(psnr_2d):
            psnr_2d_lst.append(round(float(psnr_2d), 3))
        else:
            psnr_2d_lst.append(0)

    # calc psnr over entire volume. ignore zero (i.e. -inf) entries
    psnr_3d_calc = [ss for ss in psnr_2d_lst if ss > 0]
    psnr_3d = round(sum(psnr_3d_calc) / len(psnr_3d_calc), 3)

    #mse_ = 10000 * mse(img_tgt, img_out)
    #nmse_ = nmse(img_tgt, img_out)
    #pix_val_max = img_out.max()
    #pix_val_min = img_out.min()

    img_out_ = torch.from_numpy(np.array([[img_out]]))
    img_tgt_ = torch.from_numpy(np.array([[img_tgt]]))
    
    if msssim:
        msssim_ = round(float(msssim(img_out_, img_tgt_)), 3)
    else:
        msssim_ = 0

    if hfen:
        hfen_ = round(10000 * float(hfen(img_tgt_.float(), img_out_.float())), 3)
    else:
        hfen_ = 0

    scores_dict = {'vif': vif_, 'msssim': msssim_, 'ssim': ssim_,
                   'psnr_3d': psnr_3d, 'psnr_2d_lst': psnr_2d_lst}

    return scores_dict


def norm_imgs(img_tgt, img_out, range_=0.1):
    ''' first, normalize ground-truth img_tgt to be on range [0,range_] 
        second, normalize predicted img_out according to range of img_tgt '''
    
    # must shift to zero-mean when adjusting std, then shift back to non-zero
    mu, sig = img_tgt.mean(), img_tgt.std()
    C = range_ / img_tgt.max()
    img_tgt = (img_tgt - mu) / sig
    img_tgt *= (C*sig)
    img_tgt += (C*mu)
    
    img_out = (img_out - img_out.mean()) / img_out.std()
    img_out *= img_tgt.std()
    img_out += img_tgt.mean()
    
    return img_tgt, img_out


def scale_0_1(arr):
    ''' given any array, map it to [0,1] range '''
    return (arr - arr.min()) * (1. / arr.max())


##############################################################
# below contains wrapper functions for computing metrics 
##############################################################


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    ''' compute structural similarity index metric (ssim) 
        NOTE: can get higher values by using data_range=gt.max() '''
    #return structural_similarity(gt, pred, multichannel=False, data_range=pred.max())
    return structural_similarity(gt, pred, channel_axis=False, 
                                 data_range=pred.max())


def vifp_mscale(ref, dist, sigma_nsq=1, eps=1e-10):
    ''' from https://github.com/aizvorski/video-quality/blob/master/vifp.py
        ref: reference ground-truth image
        dist: distorted image to evaluate
        sigma_nsq: ideally tune this according to input pixel values

        NOTE: order of ref, dist is important '''

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    return vifp


def get_hist(distances, bins, weights):
    ''' wrapper to get histogram values and bin edges '''
    hist, bin_edges = np.histogram(distances, bins=bins, weights=weights)
    return hist, bin_edges


def compute_frc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0
):
    ''' computes the fourier ring correlation of two 2d images
    confirmed: very similar output to compute_fsc(). 

    NOTE: given outputs, can generate frc plot via 
          plt.plot(bin_edges[:-1], density) '''
    
    image_1, image_2 = np.array(image_1), np.array(image_2)
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)

    f1, f2 = np.fft.fft2(image_1), np.fft.fft2(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1)**2, np.abs(f2)**2

    nx, ny = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))

    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()

    for xi, yi in np.array(np.meshgrid(x,y)).T.reshape(-1, 2):
        distances.append(np.sqrt(xi**2 + xi**2))
        xi = int(xi)
        yi = int(yi)
        wf1f2.append(af1f2[xi, yi])
        wf1.append(af1_2[xi, yi])
        wf2.append(af2_2[xi, yi])

    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2), bin_width)
    
    f1f2_r, _ = get_hist(distances, bins, wf1f2)
    f12_r, _ = get_hist(distances, bins, wf1)
    f22_r, bin_edges = get_hist(distances, bins, wf2)
    
    density = f1f2_r / np.sqrt(f12_r * f22_r)

    return density, bin_edges


def calc_fsc(rho1, rho2, side):
    ''' calculate Fourier Shell Correlation between two electron density maps. '''

    assert rho1.shape == rho2.shape

    df = 1.0/side
    dim_x = rho1.shape[0]
    dim_y = rho1.shape[1]
    dim_z = rho1.shape[2]
    qx_ = np.fft.fftfreq(dim_x) * dim_x * df
    qy_ = np.fft.fftfreq(dim_y) * dim_y * df
    qz_ = np.fft.fftfreq(dim_z) * dim_z * df

    qx, qy, qz = np.meshgrid(qx_,qy_,qz_,indexing='ij')
    qx_max = max(qx.max(), qy.max(), qz.max())
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1) # [0. .. 110.]

    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1

    F1 = np.fft.fftn(rho1)
    F2 = np.fft.fftn(rho2)
    numerator = scipy.ndimage.sum(np.real(F1*np.conj(F2)),
                                  labels=qbin_labels,
                                  index=np.arange(0,qbin_labels.max()+1))

    term1 = scipy.ndimage.sum(np.abs(F1)**2, labels=qbin_labels,
        index=np.arange(0,qbin_labels.max()+1))
    term2 = scipy.ndimage.sum(np.abs(F2)**2, labels=qbin_labels,
        index=np.arange(0,qbin_labels.max()+1))
    denominator = (term1*term2)**0.5

    FSC = numerator/denominator
    qidx = np.where(qbins<qx_max)

    return  np.vstack((qbins[qidx],FSC[qidx])).T


def plot_fsc(pred_rho_list, ref_rho,
             title='', label_list=None,
             pixel_spacing=1.0, figsize=(6,8)):
    ''' plot fourier shell correlation given
        pred_rho_list: list of prediction volumes
        ref_rho:       reference volume
        **args:        plot utils '''
    
    side = pixel_spacing
    refside = pixel_spacing
    assert side == refside
        
    if not label_list:
        label_list = len(pred_rho_list) * ['']
    color_list = ['bo', 'go', 'ro', 'yo', 'co']
    
    fig = plt.figure(figsize=figsize)
    
    for idx, pred_rho in enumerate(pred_rho_list):
        
        fsc = calc_fsc(pred_rho, ref_rho, side)

        x = np.linspace(fsc[0,0], fsc[-1,0], 100)
        y = np.interp(x, fsc[:,0], fsc[:,1])
        resi = np.argmin(y>=0.5)
        resx = np.interp(0.5, [y[resi+1], y[resi]], [x[resi+1], x[resi]])
        resn = round(float(1. / resx), 1)
        
        plt.plot(fsc[:,0],fsc[:,0]*0+0.5,'k--')#, figsize=(figsize,figsize))
        plt.plot(fsc[:,0],fsc[:,1],color_list[idx])#'o')
        plt.plot(x,y,'k-')
        label_ = '{}, res {} {}'.format(label_list[idx], resn, '$\mathrm{\AA}$')
        plt.plot([resx], [0.5], color_list[idx], label=label_)
        plt.legend()

    plt.xlabel('Resolution (1/$\mathrm{\AA}$)')
    plt.ylabel('Fourier Shell Correlation')
    plt.title(title)
    plt.show()

    return x, y


##############################################################
# below contains code for the hfen metric 
##############################################################


class HFENLoss(nn.Module): # Edge loss with pre_smooth
    """Calculates high frequency error norm (HFEN) between target and
     prediction used to quantify the quality of reconstruction of edges
     and fine features.
     Uses a rotationally symmetric LoG (Laplacian of Gaussian) filter to
     capture edges. The original filter kernel is of size 15×15 pixels,
     and has a standard deviation of 1.5 pixels.
     ks = 2 * int(truncate * sigma + 0.5) + 1, so use truncate=4.5
     HFEN is computed as the norm of the result obtained by LoG filtering the
     difference between the reconstructed and reference images.
    [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning, 2011
        https://ieeexplore.ieee.org/document/5617283
    [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016
        https://www.hindawi.com/journals/cmmm/2016/7571934/
    Parameters
    ----------
    img1 : torch.Tensor or torch.autograd.Variable
        Predicted image
    img2 : torch.Tensor or torch.autograd.Variable
        Target image
    norm: if true, follows [2], who define a normalized version of HFEN.
        If using RelativeL1 criterion, it's already normalized.
    """
    def __init__(self, loss_f=torch.nn.MSELoss(), kernel='log', kernel_size=15, sigma = 2.5, norm = True): #1.4 ~ 1.5
        super(HFENLoss, self).__init__()
        # can use different criteria
        self.criterion = loss_f
        self.norm = norm
        #can use different kernels like DoG instead:
        if kernel == 'dog':
            kernel = get_dog_kernel(kernel_size, sigma)
        else:
            kernel = get_log_kernel(kernel_size, sigma)
        self.filter = load_filter(kernel=kernel, kernel_size=kernel_size)

    def forward(self, img1, img2):
        self.filter.to(img1.device)
        # HFEN loss
        log1 = self.filter(img1)
        log2 = self.filter(img2)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= img2.norm()
        return hfen_loss


def get_log_kernel_5x5():
    '''
    This is a precomputed LoG kernel that has already been convolved with
    Gaussian, for edge detection.
    http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
    The 2-D LoG can be approximated by a 5 by 5 convolution kernel such as:
    This is an approximate to the LoG kernel with kernel size 5 and optimal
    sigma ~6 (0.590155...). '''
    return torch.Tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -16, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0]
            ])

#dim is the image dimension (2D=2, 3D=3, etc), but for now the final_kernel is hardcoded to 2D images
#Not sure if it would make sense in higher dimensions
#Note: Kornia suggests their laplacian kernel can also be used to generate LoG kernel:
# https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
def get_log_kernel2d(kernel_size=5, sigma=None, dim=2): #sigma=0.6; kernel_size=5

    #either kernel_size or sigma are required:
    if not kernel_size and sigma:
        kernel_size = get_kernel_size(sigma)
        kernel_size = [kernel_size] * dim #note: should it be [kernel_size] or [kernel_size-1]? look below
    elif kernel_size and not sigma:
        sigma = get_kernel_sigma(kernel_size)
        sigma = [sigma] * dim

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size-1] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    grids = torch.meshgrid([torch.arange(-size//2,size//2+1,1) for size in kernel_size])

    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, grids):
        kernel *= torch.exp(-(mgrid**2/(2.*std**2)))

    #TODO: For now hardcoded to 2 dimensions, test to make it work in any dimension
    final_kernel = (kernel) * ((grids[0]**2 + grids[1]**2) - (2*sigma[0]*sigma[1])) * (1/((2*math.pi)*(sigma[0]**2)*(sigma[1]**2)))

    #TODO: Test if normalization has to be negative (the inverted peak should not make a difference)
    final_kernel = -final_kernel / torch.sum(final_kernel)

    return final_kernel

def get_log_kernel(kernel_size: int = 5, sigma: float = None, dim: int = 2):
    '''
        Returns a Laplacian of Gaussian (LoG) kernel. If the kernel is known, use it,
        else, generate a kernel with the parameters provided (slower)
    '''
    if kernel_size ==5 and not sigma and dim == 2:
        return get_log_kernel_5x5()
    else:
        return get_log_kernel2d(kernel_size, sigma, dim)


def load_filter(kernel, kernel_size=3, in_channels=1, out_channels=1,
                stride=1, padding=True, groups=1, dim: int =2,
                requires_grad=False):
    '''
        Loads a kernel's coefficients into a Conv layer that
            can be used to convolve an image with, by default,
            for depthwise convolution
        Can use nn.Conv1d, nn.Conv2d or nn.Conv3d, depending on
            the dimension set in dim (1,2,3)
        #From Pytorch Conv2D:
            https://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
            When `groups == in_channels` and `out_channels == K * in_channels`,
            where `K` is a positive integer, this operation is also termed in
            literature as depthwise convolution.
             At groups= :attr:`in_channels`, each input channel is convolved with
             its own set of filters, of size:
             :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.
    '''

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel_conv_w(kernel, in_channels)
    assert(len(kernel.shape)==4 and kernel.shape[0]==in_channels)

    # create filter as convolutional layer
    if dim == 1:
        conv = nn.Conv1d
    elif dim == 2:
        conv = nn.Conv2d
    elif dim == 3:
        conv = nn.Conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported for convolution. \
            Received {}.'.format(dim)
        )

    filter = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        groups=groups, bias=False)
    filter.weight.data = kernel
    filter.weight.requires_grad = requires_grad
    return filter


def kernel_conv_w(kernel, channels: int =3):
    '''
        Reshape a H*W kernel to 2d depthwise convolutional
            weight (for loading in a Conv2D)
    '''

    # Dynamic window expansion. expand() does not copy memory, needs contiguous()
    kernel = kernel.expand(channels, 1, *kernel.size()).contiguous()
    return kernel


def compute_padding(kernel_size):
    '''
        Computes padding tuple. For square kernels, pad can be an
         int, else, a tuple with an element for each dimension
    '''
    # 4 or 6 ints:  (padding_left, padding_right, padding_top, padding_bottom)
    if isinstance(kernel_size, int):
        return kernel_size//2
    elif isinstance(kernel_size, list):
        computed = [k // 2 for k in kernel_size]

        out_padding = []

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
            # for even kernels we need to do asymetric padding
            if kernel_size[i] % 2 == 0:
                padding = computed_tmp - 1
            else:
                padding = computed_tmp
            out_padding.append(padding)
            out_padding.append(computed_tmp)
        return out_padding
