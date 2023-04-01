"""
Code for paper "The Predictive Forward-Forward Algorithm" (Ororbia & Mali, 2022)

################################################################################
Fits a multi-modal/mixture prior to the space of a trained PFF-RNN
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

from data_utils import DataLoader
from pff_rnn import PFF_RNN

# import general simulation utilities
from ngclearn.density.gmm import GMM

################################################################################

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

disable_prior = 0
data_dir = "../data/"
model_dir = "../exp/"
split = "train"
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["data_dir=","model_dir=","gpu_id=","split=",
                                                      "disable_prior="])
# Collect arguments from argv
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--data_dir"):
        data_dir = arg.strip()
    elif opt in ("--model_dir"):
        model_dir = arg.strip()
    elif opt in ("--split"):
        split = arg.strip()
    elif opt in ("--disable_prior"):
        disable_prior = int(arg.strip())
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())

gmm_fname = "{}/prior.gmm".format(model_dir)
latent_fname = "{}/latents.npy".format(model_dir)
batch_size = 400 #200 #100 #50 #1000 #500

mid = gpu_id
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'


xfname = "{}{}X.npy".format(data_dir,split)
X = ( tf.cast(np.load(xfname),dtype=tf.float32) )
if len(X.shape) > 2:
    X = tf.reshape(X, [X.shape[0], X.shape[1] * X.shape[2]])
x_dim = X.shape[1]
max_val = float(tf.reduce_max(X))
if max_val > 1.0:
    X = X/max_val
print("X.shape = ",X.shape)

yfname = "{}{}Y.npy".format(data_dir,split)
Y = ( tf.cast(np.load(yfname),dtype=tf.float32) )
if len(Y.shape) == 1:
    print("y_init.shape = ",Y.shape)
    nC = 10 #Y.shape[1]
    Y = tf.one_hot(Y.numpy().astype(np.int32), depth=nC)
    print("y_post.shape = ",Y.shape)
elif Y.shape[1] == 1:
    print("y_init.shape = ",Y.shape)
    nC = 10 #Y.shape[1]
    Y = tf.one_hot(tf.squeeze(Y).numpy().astype(np.int32), depth=nC)
    print("y_post.shape = ",Y.shape)
y_dim = Y.shape[1]

dataset = DataLoader(design_matrices=[("x",X.numpy()),("y",Y.numpy())],
                     batch_size=batch_size, disable_shuffle=True)

def calc_latent_map(agent, dataset, debug=False):
    z = None
    N = 0.0
    L_r = 0.0
    for batch in dataset:
        _, x = batch[0]
        _, y = batch[1]

        z2 = agent.get_latent(x, y, K, use_y_hat=True)
        e = agent.z0_hat - x
        Li = tf.reduce_sum(tf.math.square(e) * 0.5)
        L_r = Li + L_r

        if z is not None:
            z = tf.concat([z,z2],axis=0)
        else:
            z = z2
        N += x.shape[0]
    L_r = L_r / N
    return z, L_r


print("----")
with tf.device(gpu_tag):
    print(" >> Loading model from: ",model_dir)
    agent = PFF_RNN(model_dir=model_dir)
    agent.z1 = None
    agent.z0_hat = None
    K = agent.K

    z_map, Lr = calc_latent_map(agent, dataset)
    print("MSE: ",float(Lr))

    # save latent space map to disk
    print(" > Saving latents to disk: lat.shape = ",z_map.shape)
    np.save(latent_fname, z_map)

    max_w = -10000.0
    min_w = 10000.0
    max_w = max(max_w, float(tf.reduce_max(z_map)))
    min_w = min(min_w, float(tf.reduce_min(z_map)))
    print("max_z = ", max_w)
    print("min_z = ", min_w)

    if disable_prior == 0:
        print(" > Estimating latent density...")
        n_comp = 10 # number of compoments that will define the prior P(z)
        lat_density = GMM(k=n_comp)
        lat_density.fit(z_map)

        print(" > Saving density estimator to: {0}".format("gmm.pkl"))
        fd = open("{0}".format(gmm_fname), 'wb')
        pickle.dump(lat_density, fd)
        fd.close()
