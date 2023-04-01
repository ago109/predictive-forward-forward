#!/bin/bash
N_TRIALS=1 # 2

echo " >>> Running MNIST simulation!"
DATA_DIR="../data/mnist/"
OUT_DIR="../exp/pff/mnist/"
python sim_train.py --data_dir=$DATA_DIR --gpu_id=0 --n_trials=$N_TRIALS --out_dir=$OUT_DIR

echo " >>> Running K-MNIST simulation!"
DATA_DIR="../data/kmnist/"
OUT_DIR="../exp/pff/kmnist/"
python sim_train.py --data_dir=$DATA_DIR --gpu_id=0 --n_trials=$N_TRIALS --out_dir=$OUT_DIR
