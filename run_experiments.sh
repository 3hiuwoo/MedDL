#!/bin/bash

# This script runs experiments for several seeds, the pretrain and finetune dataset can be modified.
logdir=logs
pdataset=chapman
fdataset=chapman
method=clocs

for seed in {41..45}
do
    # comment out this line if the pretrained weight is already available
    python train_${method}.py --data=$pdataset --logdir=$logdir --seed=$seed
done