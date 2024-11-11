import argparse
import os
import sys
import threading
import time
import torch

import constants
from dataset import CryoLoad
from parser import set_args


def main():

    ''' TODO: delete rm pt command, clamped option in thread_{u,d} '''
    
    # define basics
    #os.system('rm -r /home/vanveen/coord_cryo_et/logs/p2s/c99')
    RECON_SUBSET = False # recon subset of volume (single chunk) for rapid prototyping
    BYPASS_TB = False # bypass tensorboard logging
    dataset, case_id_lst = get_bash_configs()

    # check gpu availability
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError('this script requires at least one gpu.')
    if n_gpus > 2:
        print("this script parallelizes across two gpu's, but you have "
              f"{n_gpus} available. modifying script will improve recon speed.")

    for case_id in case_id_lst:


        # set output directory based on dataset, case_id
        subdir = 'out_bypass_tb' if BYPASS_TB else 'out'
        dir_out = os.path.join(constants.DIR_PROJECT, subdir, dataset, f'c{case_id}')

        # set gpu indices
        gpu_u = 0
        gpu_d = 1 if n_gpus >= 2 else 0

        # set y-chunk indices
        sy_mid = constants.CONFIGS_DATASET[dataset]['img_size_load'][1] // 2 
        sy_end = int(2*sy_mid) # penultimate y chunk (upward)

        ''' define commands to call recon_chunk.py
            cmd_mid: recon middle chunk to obtain a set of init weights
            cmd_adj: recon adjacent chunk w pre-init weights from middle chunk.
                     this will be our true starting point, as we want all
                     output chunks to be reconstructed via pre-init weights
            cmd_mv:  move middle chunk to throwaway no-init directory
        '''
        cmd_base = (
            f'python recon_chunk.py --case_id {case_id} '
                                  f'--dataset {dataset} '
        )
        if BYPASS_TB:
            cmd_base += '--bypass_tb '
        cmd_mid = f'{cmd_base} --expmt_type mid --gpu {gpu_d}'
        cmd_adj = get_cmd(cmd_base, idx_y=sy_mid-2, gpu=gpu_d, dir_out=dir_out, dirn='down')
        sy_mid_str = str(sy_mid).zfill(4)
        cmd_mv = f'mv {dir_out}/sy{sy_mid_str} {dir_out}/sy{sy_mid_str}_no-init'

        # define parallel threads: moving up, down from middle chunk
        # begin each by loading pre-init model weights from adjacent chunk
        thread_u = threading.Thread(target=go_up,
                                    args=(cmd_base, sy_mid, sy_end, gpu_u, dir_out))
                                    #args=(cmd_base, sy_mid, sy_mid+2, gpu_u, dir_out))
        thread_d = threading.Thread(target=go_down,
                                    args=(cmd_base, sy_mid-4, 0, gpu_d, dir_out))
                                    #args=(cmd_base, sy_mid-4, sy_mid-6, gpu_d, dir_out))

        t0 = time.time() # begin recons

        os.system(cmd_mid) # reconstruct middle chunk
        if RECON_SUBSET:
            continue
        
        os.system(cmd_adj) # reconstruct adjacent downward chunk
        os.system(cmd_mv) # move middle chunk to no-init directory

        if n_gpus == 1: # do threads in series: first up, then down from middle
            thread_u.start()
            thread_u.join()
            thread_d.start()
            thread_d.join()
        if n_gpus >= 2: # do threads in parallel, then wait for both to finish
            thread_u.start()
            thread_d.start()
            thread_u.join()
            thread_d.join()

        # record time to reconstruct volume
        t_finish = str(round((time.time() - t0) / 60, 3))
        print(f'reconstructed volume in {t_finish} minutes')
        fn_time = os.path.join(dir_out, 'time.txt')
        if not os.path.exists(fn_time):
            with open(fn_time, 'w') as f:
                f.write(t_finish)

        # save output volume to mrc file
        wrap_save_to_mrc(dataset, case_id, BYPASS_TB)
        


def get_bash_configs():

    parser = argparse.ArgumentParser(description="main script command line")
    parser.add_argument('--dataset', type=str, required=True,
                        help='cryoet dataset')
    parser.add_argument('--case_id_lst', nargs='*', type=int, required=False,
                        help='list of case_id expmts')
    tmp = parser.parse_args()

    # if case_id not specified, use default case_id 0
    if len(tmp.case_id_lst) == 0:
        tmp.case_id_lst = [0]

    return tmp.dataset, tmp.case_id_lst


def get_cmd(
        cmd_base, # (str): base command
        idx_y, # (int): index of y-chunk
        gpu, # (int): 0 or 1, i.e. gpu id
        dir_out, # (str): directory for expmt output
        dirn, # (str): 'up' or 'down', i.e. direction we're moving in
):
    ''' get command string for running over particular chunk '''

    # get path of initializing model
    sy_init = idx_y - 2 if dirn == 'up' else idx_y + 2 if dirn == 'down' else None
    path_init_model = os.path.join(dir_out, 
                                   f'sy{str(sy_init).zfill(4)}',
                                   'checkpoints/model_final.pth')
    
    cmd_chunk = (
        f'{cmd_base} --expmt_type all --gpu {gpu} '
        f'--idx_y {idx_y} --path_init_model {path_init_model}'
    )

    return cmd_chunk


def go_up(cmd_base, sy_begin, sy_end, gpu, dir_out):
    ''' recon from sy_begin to sy_end chunks going up '''
    for idx_y in range(sy_begin, sy_end+2, 2):
        cmd_nxt = get_cmd(cmd_base, idx_y, gpu, dir_out, dirn='up')
        os.system(cmd_nxt)


def go_down(cmd_base, sy_begin, sy_end, gpu, dir_out):
    ''' recon from sy_begin to sy_end chunks going down '''
    for idx_y in range(sy_begin, sy_end-2, -2):
        cmd_nxt = get_cmd(cmd_base, idx_y, gpu, dir_out, dirn='down')
        os.system(cmd_nxt)


def wrap_save_to_mrc(dataset, case_id, bypass_tb=False):
    ''' instantiate CryoLoad object so that output is saved to mrc '''
    args = argparse.ArgumentParser()
    args.dataset = dataset
    args.case_id = case_id
    args.bypass_tb = bypass_tb
    args = set_args(args, purpose='load')
    _ = CryoLoad(args)

    dir_indiv = os.path.join(args.dir_out, 'indiv_chunks')
    os.system(f'mkdir {dir_indiv}; mv {args.dir_out}/sy* {dir_indiv}')


if __name__ == '__main__':
    main()