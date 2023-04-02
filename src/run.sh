#!/bin/bash
GPU_ID=0
N_TRIALS=2 # number of experimental trials to run

echo " >>> Running MNIST simulation!"
DATA_DIR="../data/mnist/"
OUT_DIR="../exp/pff/mnist/"
python sim_train.py --data_dir=$DATA_DIR --gpu_id=$GPU_ID --n_trials=$N_TRIALS --out_dir=$OUT_DIR

echo " >>> Running K-MNIST simulation!"
DATA_DIR="../data/kmnist/"
OUT_DIR="../exp/pff/kmnist/"
python sim_train.py --data_dir=$DATA_DIR --gpu_id=$GPU_ID --n_trials=$N_TRIALS --out_dir=$OUT_DIR
