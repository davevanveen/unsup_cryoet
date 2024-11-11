#!/bin/bash

####################
### define arguments

# dataset to run expmt (str)
dataset="p2s"

# case_ids defining expmt params (list of ints)
# e.g. "1 2 5" to run case_ids 1, 2, 5 in series
# if empty, will do case_id=0, i.e. default params 
case_id="99"  


####################
### pass arguments to the python script
python recon_volume.py --dataset "$dataset" \
                       --case_id $case_id

