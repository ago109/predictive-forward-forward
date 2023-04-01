"""
Code for paper "The Predictive Forward-Forward Algorithm" (Ororbia & Mali, 2022)
"""

import os
import sys, getopt, optparse
import pickle
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np

import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet
#cmap = plt.cm.cividis # red-green color blind friendly palette

################################################################################

# GPU arguments
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["data_dir=","model_dir=","gpu_id=","split="])

# Collect arguments from argv
model_dir = "../exp/"
data_dir = "../data/"
use_tsne = True # (args.getArg("use_tsne").lower() == 'true')
split = "test"

use_gpu = False
gpu_id = -1
for opt, arg in options:
    if opt in ("--data_dir"):
        data_dir = arg.strip()
    elif opt in ("--model_dir"):
        model_dir = arg.strip()
    elif opt in ("--split"):
        split = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True

mid = gpu_id
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

plot_fname = "{}/lat_viz.jpg".format(model_dir)
latents_fname = "{}/latents.npy".format(model_dir)

#batch_size = int(args.getArg("batch_size")) #128
delimiter = "\t"
# xfname = args.getArg("xfname") #"../data/mnist/trainX.tsv"
yfname = "{}/{}Y.npy".format(data_dir,split)
Y = tf.cast(np.load(yfname),dtype=tf.float32).numpy()

with tf.device(gpu_tag):

    z_lat = tf.cast(np.load(latents_fname),dtype=tf.float32)
    print("Lat.shape = {}".format(z_lat.shape))
    y_sample = Y
    if len(y_sample.shape) == 1:
        print("y_init.shape = ",y_sample.shape)
        nC = 10 #Y.shape[1]
        y_sample = tf.one_hot(tf.cast(y_sample,dtype=tf.float32).numpy().astype(np.int32), depth=nC)
        print("y_post.shape = ",y_sample.shape)
    elif y_sample.shape[1] == 1:
        print("y_init.shape = ",y_sample.shape)
        nC = 10 #Y.shape[1]
        y_sample = tf.one_hot(tf.squeeze(y_sample).numpy().astype(np.int32), depth=nC)
        print("y_post.shape = ",y_sample.shape)

    max_w = -10000.0
    min_w = 10000.0
    max_w = max(max_w, float(tf.reduce_max(z_lat)))
    min_w = min(min_w, float(tf.reduce_min(z_lat)))
    print("max_z = ", max_w)
    print("min_z = ", min_w)
    print("Y.shape = ",y_sample.shape)

    z_top_dim = z_lat.shape[1]
    z_2D = None
    if z_top_dim != 2:
        from sklearn.decomposition import IncrementalPCA
        print(" > Projecting latents via iPCA...")
        if use_tsne is True:
            n_comp = 32 #10 #16 #50
            if z_lat.shape[1] < n_comp:
                n_comp = z_lat.shape[1] - 2 #z_top.shape[1]-2
                n_comp = max(2, n_comp)
            ipca = IncrementalPCA(n_components=n_comp, batch_size=50)
            ipca.fit(z_lat.numpy())
            z_2D = ipca.transform(z_lat.numpy())
            print("PCA.lat.shape = ",z_2D.shape)
            print(" > Finishing projection via t-SNE...")
            from sklearn.manifold import TSNE
            z_2D = TSNE(n_components=2,perplexity=30).fit_transform(z_2D)
            #z_2D.shape
        else:
            ipca = IncrementalPCA(n_components=2, batch_size=50)
            ipca.fit(z_lat.numpy())
            z_2D = ipca.transform(z_lat.numpy())
    else:
        z_2D = z_lat

    print(" > Plotting 2D latent encodings...")
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2D[:, 0], z_2D[:, 1], c=np.argmax(y_sample, 1), cmap=cmap)
    plt.colorbar()
    plt.grid()
    plt.savefig("{0}".format(plot_fname), dpi=300) # latents.jpg
    plt.clf()
