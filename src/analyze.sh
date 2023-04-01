#!/bin/bash
GPU_ID=0
MODEL_DIR="../exp/pff/mnist/trial0/"
DATA_DIR="../data/mnist/"

# for prior distribution fitting / sampling
echo " ---------- Fitting Prior ---------- "
python fit_gmm.py --data_dir=$DATA_DIR --gpu_id=$GPU_ID --split=train --model_dir=$MODEL_DIR
#python plot_tsne.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=train
echo " ---------- Sampling Model ---------- "
python eval_model.py --gpu_id=$GPU_ID --model_dir=$MODEL_DIR

# for latent code visualization
echo " ---------- Extracting Test Latents ---------- "
python fit_gmm.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=test --disable_prior=1
echo " ---------- Visualizing Test Latents ---------- "
python plot_tsne.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gpu_id=$GPU_ID --split=test
