import torch


def tv_loss(args, img):
    ''' total variation regularization loss '''
   
    if args.lam_tv == 0 or not args.lam_tv:
        return {'tv_loss': torch.tensor(0.)}
  
    assert img.ndim == 5 # [bs, x, y, z, 1]
    assert img.shape[0] == 1 # can't do this w non-adjacent slices
    bs, len_x, len_y, len_z, _ = img.size()
    img = img.float() # otw if float16, get overflow on torch.pow() operation
    
    tv_x = torch.pow(img[:, 1:, :, :, :]-img[:, :-1, :, :, :], 2).sum()
    tv_y = torch.pow(img[:, :, 1:, :, :]-img[:, :, :-1, :, :], 2).sum()
    tv_z = torch.pow(img[:, :, :, 1:, :]-img[:, :, :, :-1, :], 2).sum()

    loss = (tv_x + tv_y + tv_z) / (bs * len_x * len_y * len_z)

    loss *= args.lam_tv

    return {'tv_loss': loss}


def function_mse(model_output, gt):
    idx = model_output['model_in']['idx'].long().squeeze()
    loss = (model_output['model_out']['output'][:, idx] - gt['func'][:, idx]) ** 2
    return {'l2_loss': loss.mean()}


def image_mse(model_output, gt):
    return l2_loss(model_output, gt)


def l2_loss(model_output, gt):
    
    if type(model_output) is dict and type(gt) is dict:
        if 'complex' in model_output['model_out']:
            c = model_output['model_out']['complex']
            loss = (c.real - gt['img']) ** 2
            imag_loss = (c.imag) ** 2
            return {'l2_loss': loss.mean(), 'imag_loss': imag_loss.mean()}

        else:
            loss = (model_output['model_out']['output'] - gt['img']) ** 2
            return {'l2_loss': loss.mean()}
    else:
        loss = (model_output - gt) ** 2
        return {'l2_loss': loss.mean()}


def noise_loss_gt(model_output, gt_denoised, lam_dn_gt=0):
    ''' compute || x - D(gt) ||_2 for denoiser D 
        where D(gt) = gt_denoised has already been denoised '''
   
    if lam_dn_gt == 0:
        return {'noise_loss': torch.tensor(0.)}

    loss = lam_dn_gt * image_mse(model_output, gt_denoised)['l2_loss']

    return {'noise_loss': loss}


def l1_loss(model_output, gt):
    
    l1 = torch.nn.L1Loss()
    
    if type(model_output) is dict and type(gt) is dict:
        if 'complex' in model_output['model_out']:
            raise NotImplementedError
        else:
            loss = l1(model_output['model_out']['output'], gt['img'])
    else:
        loss = l1(model_output, gt)

    return {'l1_loss': loss}


def multiscale_image_mse(model_output, gt, use_resized=False):
    if use_resized:
        loss = [(out - gt_img)**2 for out, gt_img in zip(model_output['model_out']['output'], gt['img'])]
    else:
        loss = [(out - gt['img'])**2 for out in model_output['model_out']['output']]

    loss = torch.stack(loss).mean()

    return {'l2_loss': loss}


def get_loss_fn(loss_fn):
    ''' wrapper to call loss function via config file '''
    if loss_fn == 'l1':
        return l1_loss
    elif loss_fn == 'l2':
        return l2_loss
    else:
        raise NotImplementedError