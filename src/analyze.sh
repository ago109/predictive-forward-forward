#!/bin/bash
GPU_ID=0
N_TRIALS=2

## Analyze MNIST
MODEL_TOPDIR="../exp/pff/mnist/"
MODEL_DIR="../exp/pff/mnist/trial0/" ## <-- choose a specific trial idx to analyze
DATA_DIR="../data/mnist/"

echo " ---------- Evaluating MNIST Test Performance ---------- "
python eval_model.py --data_dir=$DATA_DIR --split=test --model_topdir=$MODEL_TOPDIR --gpu_id=$GPU_ID --n_trials=$N_TRIALS --out_dir=$MODEL_TOPDIR

# for prior distribution fitting / sampling
echo " ---------- Fitting MNIST Prior ---------- "
python fit_gmm.py --data_dir=$DATA_DIR --gpu_id=$GPU_ID --split=train --model_dir=$MODEL_DIR
echo " ---------- Sampling MNIST Model ---------- "
python sample_model.py --gpu_id=$GPU_ID --model_dir=$MODEL_DIR

# for latent code visualization
echo " ---------- Extracting MNIST Test Latents ---------- "
python fit_gmm.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=test --disable_prior=1
echo " ---------- Visualizing MNIST Test Latents ---------- "
python plot_tsne.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=test


## Analyze K-MNIST
MODEL_TOPDIR="../exp/pff/kmnist/"
MODEL_DIR="../exp/pff/kmnist/trial0/" ## <-- choose a specific trial idx to analyze
DATA_DIR="../data/kmnist/"

echo " ---------- Evaluating K-MNIST Test Performance ---------- "
python eval_model.py --data_dir=$DATA_DIR --split=test --model_topdir=$MODEL_TOPDIR --gpu_id=$GPU_ID --n_trials=$N_TRIALS --out_dir=$MODEL_TOPDIR

# for prior distribution fitting / sampling
echo " ---------- Fitting K-MNIST Prior ---------- "
python fit_gmm.py --data_dir=$DATA_DIR --gpu_id=$GPU_ID --split=train --model_dir=$MODEL_DIR
echo " ---------- Sampling K-MNIST Model ---------- "
python sample_model.py --gpu_id=$GPU_ID --model_dir=$MODEL_DIR

# for latent code visualization
echo " ---------- Extracting K-MNIST Test Latents ---------- "
python fit_gmm.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=test --disable_prior=1
echo " ---------- Visualizing K-MNIST Test Latents ---------- "
python plot_tsne.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=test
