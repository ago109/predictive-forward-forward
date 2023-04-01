"""
Code for paper "The Predictive Forward-Forward Algorithm" (Ororbia & Mali, 2022)

################################################################################
Samples a trained PFF-RNN system.
################################################################################
"""

import os
import sys, getopt, optparse
import pickle
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import time
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#cmap = plt.cm.jet

from pff_rnn import PFF_RNN

################################################################################

def deserialize(fname): ## object "loading" routine
    fd = open(fname, 'rb')
    object = pickle.load( fd )
    fd.close()
    return object

def plot_img_grid(samples, fname, nx, ny, px, py, plt, rotNeg90=False): # rows, cols,...
    px_dim = px
    py_dim = py
    canvas = np.empty((px_dim*nx, py_dim*ny))
    ptr = 0
    for i in range(0,nx,1):
        for j in range(0,ny,1):
            #xs = tf.expand_dims(tf.cast(samples[ptr,:],dtype=tf.float32),axis=0)
            xs = np.expand_dims(samples[ptr,:],axis=0)
            #xs = xs.numpy() #tf.make_ndarray(x_mean)
            xs = xs[0].reshape(px_dim, py_dim)
            if rotNeg90 is True:
                xs = np.rot90(xs, -1)
            canvas[(nx-i-1)*px_dim:(nx-i)*px_dim, j*py_dim:(j+1)*py_dim] = xs
            ptr += 1
    plt.figure(figsize=(12, 14))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("{0}".format(fname), bbox_inches='tight', pad_inches=0)
    plt.clf()

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

model_dir = "../exp/"
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["model_dir=","gpu_id="])
# Collect arguments from argv
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--model_dir"):
        model_dir = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())

gmm_fname = "{}/prior.gmm".format(model_dir)

mid = gpu_id #0
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

def get_n_comp(gmm, use_sklearn=True):
    if use_sklearn is True:
        return gmm.n_components
    else:
        return gmm.k

def sample_gmm(gmm, n_samps, use_sklearn=False):
    if use_sklearn is True:
        np_samps, np_labs = gmm.sample(n_samps)
        z_samp = tf.cast(np_samps,dtype=tf.float32)
    else:
        z_samp, z_labs = gmm.sample(n_samps)
        np_labs = tf.squeeze(z_labs).numpy()
    y_s = tf.one_hot(np_labs, get_n_comp(gmm, use_sklearn=use_sklearn))
    return z_samp, y_s

print("----")
with tf.device(gpu_tag):
    print(" >> Loading prior P(z):  ",gmm_fname)
    prior = deserialize(gmm_fname)
    print(" >> Loading model P(x|z):  ",model_dir)
    agent = PFF_RNN(model_dir=model_dir)
    print(agent.V.shape)
    nrow = 10 #2
    ncol = 10 #4
    n_samp = nrow * ncol # per class
    print(" >> Generating confabulations from P(x|z)P(z)...")
    for _ in range(15): ## jitter the prior
        z2s, _ = sample_gmm(prior, n_samp)
    xs = agent.sample(y=z2s)
    fname = "{}samples.jpg".format(model_dir)
    plot_img_grid(xs.numpy(), fname, nx=nrow, ny=ncol, px=28, py=28, plt=plt)
    plt.close()
