import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
from functools import partial

import constants
import loss_functions as loss_fns
import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(args, model, data,
          loss_fn, summary_fn, ckpt_epoch=0):
   
    ### preliminaries #########################################################
    # directories, tensorboard writer
    summ_dir = os.path.join(args.dir_out, 'summaries')
    ckpt_dir = os.path.join(args.dir_out, 'checkpoints')
    val_metrics_dir = os.path.join(ckpt_dir, 'val_metrics')
    for dir_ in [summ_dir, ckpt_dir, val_metrics_dir]:
        utils.cond_mkdir(dir_)
    writer = SummaryWriter(summ_dir)
    
    # optimizer, dataloader 
    optim = torch.optim.AdamW(lr=args.lr, params=model.parameters(),
                              amsgrad=True, weight_decay=0)
    dataloader = DataLoader(data, batch_size=data.batch_size,
                            shuffle=True, pin_memory=True)
    trn_generator = iter(dataloader) 
    scaler = torch.cuda.amp.GradScaler() # for mixed precision training
    
    # learning rate scheduler
    if constants.USE_LR_SCHEDULER:
        optim.param_groups[0]['lr'] = 1
        scheduler = partial(lr_log_schedule, args=args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=scheduler)
    else:
        scheduler = None

    ### main training loop ####################################################
    with tqdm(total=args.n_epochs) as pbar:
        trn_losses = []
        for epoch in range(args.n_epochs):
            curr_epoch = epoch + ckpt_epoch

            # checkpoint to save model, losses
            if not args.bypass_tb and (not epoch % constants.EPOCHS_TIL_CKPT and epoch):
                save_model(model, ckpt_dir, 'model_epoch_%04d.pth' % (curr_epoch))
                save_scalars(trn_losses, ckpt_dir, 
                             'trn_losses_epoch_%04d.txt' % (curr_epoch))
                
            try: # iterate thru dataloader
                inp, tgt = next(trn_generator)
            except StopIteration:
                trn_generator = iter(dataloader)
                inp, tgt = next(trn_generator)
            inp, tgt = dict2cuda(inp), dict2cuda(tgt)
             
            t0 = time.perf_counter()

            for step in range(args.steps_per_epoch): # loop over each batch in epoch
                
                # compute model output over one batch 
                inp_b = inp['coords'][step][None, :] # input_batch
                
                with torch.cuda.amp.autocast(enabled=constants.USE_HALF_PRECISION):
                    out = model(inp_b)['model_out']['output'] # [1,x,y,z,1]
                
                tgt_prj = tgt['prj'][step][None, :] # [1,61,x,y]
                trn_prj = compute_prj(out, data.ct_projector, args)
                
                with torch.cuda.amp.autocast(enabled=constants.USE_HALF_PRECISION):
                    losses = loss_fn(trn_prj, tgt_prj)

                loss_tv = loss_fns.tv_loss(args, out)
                #loss_fft = loss_fns.fft_loss(args, out, data.mask)
                losses = {**losses, **loss_tv}#, **loss_fft} # combine dicts

                # compute trn loss. TODO: avg across all batches in epoch
                trn_loss = 0.
                for loss_name, loss in losses.items():
                    if True: #not args.bypass_tb:
                        writer.add_scalar(loss_name, loss.mean(), epoch)
                    trn_loss += loss.mean() 
                trn_losses.append(trn_loss.item()) # NOTE: update xx if step-wise
                writer.add_scalar('total_trn_loss', trn_loss, epoch)
                
                # backprop over avg loss of all batches
                optim.zero_grad(set_to_none=True)
                scaler.scale(trn_loss).backward() # old: trn_loss.backward()

            # update tensorboard summary, save val metrics if applicable
            if False: #not epoch % constants.EPOCHS_TIL_SUMMARY:
                save_model(model, ckpt_dir, 'model_current.pth')
                summary_fn(model, inp, tgt, writer, epoch, args, data.is_sim)

            # update optimizer after each epoch
            scaler.step(optim) # old: optim.step()
            scaler.update()
            if True: #not args.bypass_tb:
                writer.add_scalar('lr', optim.param_groups[0]['lr'], epoch)
            if scheduler is not None:
                scheduler.step() 

            # update progress bar, calculate validation loss 
            pbar.update(1)
            if not epoch % constants.EPOCHS_TIL_SUMMARY:
                tqdm.write('epoch %d, total loss %0.6f, iteration time %0.6f' % 
                            (epoch, trn_loss, time.perf_counter() - t0))
                if False: #val_dataloader is not None:
                    calc_val_loss(model, val_dataloader, loss_fn, writer, epoch)

        print('loss {}'.format(trn_loss.detach().cpu().numpy()))
        # save final model, output 
        save_model(model, ckpt_dir, 'model_final.pth')
        save_scalars(trn_losses, ckpt_dir, 'trn_losses_final.txt')
    

def compute_prj(out_img, ct_projector, args):
    ''' given output image from the model
        compute projections for corresponding y-slices
        args
            use_batches: if gradient step over subset of y slices 
                         expand batch output by repeating values c times 
                         s.t. projector input has correct dimn. c = steps_per_epoch 
                         basically padding w repeat data to avoid dimn mismatch
                         
        TODO: note this function has some redundancy w dataset.back_proj '''

    out_img = out_img.squeeze(-1) # [1,x,y,z,1] --> [1,x,y,z]
    out_img = torch.repeat_interleave(out_img, 
                    args.steps_per_epoch, dim=2) # [1,x,y,z]-->[1,x,c*y,z]

    # do forward projection for all 90 directions
    trn_prj = ct_projector.forward_project(out_img, args)
    
    # compress repeated slice_y's. nn.MaxPool1d requires 3d tensor 
    m = torch.nn.MaxPool1d(kernel_size=args.steps_per_epoch) # [61,x,y]
    trn_prj = m(trn_prj.squeeze(0)).unsqueeze(0) # [1,61,x,y]

    return trn_prj


def save_model(model, ckpt_dir, filename):
    torch.save(model.state_dict(), os.path.join(ckpt_dir, filename))


def save_scalars(scalars, ckpt_dir, filename):
    np.savetxt(os.path.join(ckpt_dir, filename), np.array(scalars))


def calc_val_loss(model, val_dataloader, loss_fn, writer, step):
    ''' during training loop, calculate loss on validation set '''
    model.eval()
    with torch.no_grad():
        val_losses = []
        for inp, tgt in val_dataloader:
            inp = dict2cuda(inp)
            tgt = dict2cuda(tgt)
            out = model(inp)
            val_loss = loss_fn(out, tgt)
            val_loss = val_loss['func_loss'].item()
            val_losses.append(val_loss)
        # print(np.mean(val_losses))
        writer.add_scalar("val_loss", np.mean(val_losses), step)
    model.train()


def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cuda(value)})
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.cuda() for v in value]})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.cpu() for v in value]})
        else:
            tmp.update({key: value})
    return tmp


def lr_log_schedule(epoch, args):
    ''' define learning rate according to lr = alpha * exp(log_lr0 + log_lrn)
        args:
            epoch:        current epoch
            n_epochs:     number of total epochs
            lr0, lrn:     start, end learning rates
            decay_start:  epoch at which to begin decay '''

    alpha = 1

    if args.lr_decay_start > 0: # start lr decay after certain number of epochs
        if epoch < args.lr_decay_start:
            return args.lr
        elif epoch > args.lr_decay_end:
            return args.lrn
        n_decay_epochs = args.lr_decay_end - args.lr_decay_start # number of epochs to decay
    else:
        n_decay_epochs = args.n_epochs

    perc_complete = (epoch - args.lr_decay_start) / n_decay_epochs
    log_lr0 = (1 - perc_complete) * np.log(args.lr)
    log_lrn = perc_complete * np.log(args.lrn)

    return  alpha * np.exp(log_lr0 + log_lrn)