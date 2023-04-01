"""
Code for paper "The Predictive Forward-Forward Algorithm" (Ororbia & Mali, 2022)

################################################################################
Simulates the training/adaptation of a recurrent neural system composed of
a representation and generative circuit, trained via the preditive forward-forward
process.
Note that this code focuses on datasets of gray-scale images/patterns.
################################################################################
"""

import os
import sys, getopt, optparse
import pickle
#import dill as pickle
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import time
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#cmap = plt.cm.jet

# import general simulation utilities
from data_utils import DataLoader

def plot_img_grid(samples, fname, nx, ny, px, py, plt, rotNeg90=False): # rows, cols,...
    '''
    Visualizes a matrix of vector patterns in the form of an image grid plot.
    '''
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
    #print(" SAVE: {0}{1}".format(out_dir,"gmm_decoded_samples.jpg"))
    plt.savefig("{0}".format(fname), bbox_inches='tight', pad_inches=0)
    plt.clf()

from pff_rnn import PFF_RNN

################################################################################

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

out_dir = "../exp/"
data_dir = "../data/mnist/"
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["data_dir=","gpu_id=","n_trials=","out_dir="])
# Collect arguments from argv
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--data_dir"):
        data_dir = arg.strip()
    elif opt in ("--out_dir"):
        out_dir = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
    elif opt in ("--n_trials"):
        n_trials = int(arg.strip())
print(" Exp out dir: ",out_dir)

mid = gpu_id # 0
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

print(" >>> Run sim on {} w/ GPU {}".format(data_dir, mid))

xfname = "{}/trainX.npy".format(data_dir)
yfname = "{}/trainY.npy".format(data_dir)

X = ( tf.cast(np.load(xfname, allow_pickle=True),dtype=tf.float32) )
if len(X.shape) > 2:
    X = tf.reshape(X, [X.shape[0], X.shape[1] * X.shape[2]])
    print(X.shape)
x_dim = X.shape[1]
max_val = float(tf.reduce_max(X))
if max_val > 1.0:
    X = X/max_val

Y = ( tf.cast(np.load(yfname, allow_pickle=True),dtype=tf.float32) )
y_dim = Y.shape[1]
print("Y.shape = ",Y.shape)

n_iter = 60 #100 #60 # number of training iterations
batch_size = 500 # batch size
dev_batch_size = 500 # dev batch size
dataset = DataLoader(design_matrices=[("x",X.numpy()),("y",Y.numpy())], batch_size=batch_size)

print(" > Loading dev set")
xfname = "{}/validX.npy".format(data_dir)
yfname = "{}/validY.npy".format(data_dir)
Xdev = ( tf.cast(np.load(xfname, allow_pickle=True),dtype=tf.float32) )

if len(Xdev.shape) > 2:
    Xdev = tf.reshape(Xdev, [Xdev.shape[0], Xdev.shape[1] * Xdev.shape[2]])
print("Xdev.shape = ",Xdev.shape)
max_val = float(tf.reduce_max(Xdev))
if max_val > 1.0:
    Xdev = Xdev/max_val

Ydev = ( tf.cast(np.load(yfname, allow_pickle=True),dtype=tf.float32) )
if len(Ydev.shape) == 1:
    nC = Y.shape[1]
    Ydev = tf.one_hot(Ydev.numpy().astype(np.int32), depth=nC)
elif Ydev.shape[1] == 1:
    nC = 10
    Ydev = tf.one_hot(tf.squeeze(Ydev).numpy().astype(np.int32), depth=nC)
print("Ydev.shape = ",Ydev.shape)

devset = DataLoader(design_matrices=[("x",Xdev.numpy()), ("y",Ydev.numpy())],
                                     batch_size=dev_batch_size, disable_shuffle=True)

def classify(agent, x):
    K_low = int(agent.K/2) - 1 # 3
    K_high = int(agent.K/2) + 1 # 5
    x_ = x
    Ey = None
    z_lat = agent.forward(x_) # do forward init pass
    for i in range(agent.y_dim):
        z_lat_ = []
        for ii in range(len(z_lat)):
            z_lat_.append(z_lat[ii] + 0)

        yi = tf.ones([x.shape[0],agent.y_dim]) * tf.expand_dims(tf.one_hot(i,depth=agent.y_dim),axis=0)

        gi = 0.0
        for k in range(K_high):
            z_lat_, p_g = agent._step(x_, yi, z_lat_, thr=0.0)
            if k >= K_low and k <= K_high: # only keep goodness in middle iterations
                gi = ((p_g[0] + p_g[1])*0.5) + gi

        if i > 0:
            Ey = tf.concat([Ey,gi],axis=1)
        else:
            Ey = gi

    Ey = Ey / (3.0)
    y_hat = tf.nn.softmax(Ey)
    return y_hat, Ey

def eval(agent, dataset, debug=False, save_img=True, out_dir=""):
    '''
    Evaluates the current state of the agent given a dataset (data-loader).
    '''
    N = 0.0
    Ny = 0.0
    Acc = 0.0
    Ly = 0.0
    Lx = 0.0
    tt = 0
    #debug = True
    for batch in dataset:
        _, x = batch[0]
        _, y = batch[1]
        N += x.shape[0]
        Ny += float(tf.reduce_sum(y))

        y_hat, Ey = classify(agent, x)
        Ly += tf.reduce_sum(-tf.reduce_sum(y * tf.math.log(y_hat), axis=1, keepdims=True))

        z_lat = agent.forward(x)
        x_hat = agent.sample(z=z_lat[len(z_lat)-2])

        ex = x_hat - x
        Lx += tf.reduce_sum(tf.reduce_sum(tf.math.square(ex),axis=1,keepdims=True))

        if debug == True:
            print("------------------------")
            print(Ey[0:4,:])
            print(y_hat[0:4,:])
            print("------------------------")

        y_ind = tf.cast(tf.argmax(y,1),dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(Ey,1),dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred,y_ind),dtype=tf.float32) #* y_m
        Acc += tf.reduce_sum( comp )
    Ly = Ly/Ny
    Acc = Acc/Ny
    Lx = Lx/Ny

    if save_img == True:
        fname = "{}/x_samples.png".format(out_dir)
        plot_img_grid(x_hat.numpy(), fname, nx=10, ny=10, px=28, py=28, plt=plt)
        plt.close()

        fname = "{}/x_data.png".format(out_dir)
        plot_img_grid(x, fname, nx=10, ny=10, px=28, py=28, plt=plt)
        plt.close()

    return Ly, Acc, Lx

print("----")
with tf.device(gpu_tag):

    best_acc_list = []
    acc_list = []
    for tr in range(n_trials):
        acc_scores = [] # tracks acc during training w/in a trial
        ########################################################################
        ## create model
        ########################################################################
        model_dir = "{}/trial{}/".format(out_dir, tr)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        args = {"x_dim": x_dim,
                "y_dim": y_dim,
                "n_units": 2000,
                "K":12,
                "thr": 10.0,
                "eps_r": 0.01,
                "eps_g": 0.025}
        agent = PFF_RNN(args=args)
        ## set up optimization
        eta = 0.00025 # 0.0005 # for grnn
        reg_lambda = 0.0
        #reg_lambda = 0.0001 # works nice for kmnist
        opt = tf.keras.optimizers.Adam(eta)

        g_eta = 0.00025 # 0.0005 #0.001 #0.0005 #0.001
        g_reg_lambda = 0
        g_opt = tf.keras.optimizers.Adam(g_eta)
        ########################################################################

        ########################################################################
        ## begin simulation
        ########################################################################
        Ly, Acc, Lx = eval(agent, devset, out_dir=model_dir)
        acc_scores.append(Acc)
        print("{}: L {} Acc = {}  Lx {}".format(-1, Ly, Acc, Lx))

        best_Acc = Acc
        best_Ly = Ly
        for t in range(n_iter):
            N = 0.0
            Ng = 0.0
            Lg = 0.0
            Ly = 0.0
            for batch in dataset:
                _, x = batch[0]
                _, y = batch[1]
                N += x.shape[0]

                # create negative data (x, y_neg)
                x_neg = x
                y_neg = tf.random.uniform(y.shape, 0.0, 1.0) * (1.0 - y)
                y_neg = tf.one_hot(tf.argmax(y_neg,axis=1), depth=agent.y_dim)

                ## create full batch
                x_ = tf.concat([x,x_neg],axis=0)
                y_ = tf.concat([y,y_neg],axis=0)
                lab = tf.concat([tf.ones([x.shape[0],1]),tf.zeros([x_neg.shape[0],1])],axis=0)
                ## update model given full batch
                z_lat = agent.forward(x_)
                Lg_t, _, Lgen_t, x_hat = agent.infer(x_, y_, lab, z_lat, agent.K, opt, g_opt, reg_lambda=reg_lambda, g_reg_lambda=g_reg_lambda) # total goodness
                Lg_t = Lg_t * (x.shape[0] + x_neg.shape[0])

                #y_hat = y_hat[0:x.shape[0],:]
                y_hat = agent.classify(x)
                Ly_t = tf.reduce_sum(-tf.reduce_sum(y * tf.math.log(y_hat), axis=1, keepdims=True))

                ## track losses
                Ly = Ly_t + Ly
                Lg = Lg_t + Lg
                Ng += (x.shape[0] + x_neg.shape[0])

                print("\r   {}: Ly = {}  L = {} w/ {} samples".format(t, Ly/N, Lg/Ng, N), end="")
            print()
            print("--------------------------------------")

            Ly, Acc, Lx = eval(agent, devset, out_dir=model_dir)
            acc_scores.append(Acc)
            np.save("{}/dev_acc.npy".format(model_dir), np.asarray(acc_scores))
            print("{}: L {} Acc = {}  Lx {}".format(t, Ly, Acc, Lx))

            if Acc > best_Acc:
                best_Acc = Acc
                best_Ly = Ly

                print(" >> Saving model to:  ",model_dir)
                agent.save_model(model_dir)

        print("************")
        Ly, Acc, _ = eval(agent, dataset, out_dir=model_dir, save_img=False)
        print("   Train: Ly {} Acc = {}".format(Ly, Acc))
        print("Best.Dev: Ly {} Acc = {}".format(best_Ly, best_Acc))

        acc_list.append(1.0 - Acc)
        best_acc_list.append(1.0 - best_Acc)

    ############################################################################
    ## calc post-trial statistics
    n_dec = 4
    mu = round(np.mean(np.asarray(best_acc_list)), n_dec)
    sd = round(np.std(np.asarray(best_acc_list)), n_dec)
    print("  Test.Acc = {:.4f} \pm {:.4f}".format(mu, sd))

    ## store result to disk just in case...
    results_fname = "{}/post_train_results.txt".format(out_dir)
    log_t = open(results_fname,"a")
    log_t.write("Generalization Results:\n")
    log_t.write("  Test.Acc = {:.4f} \pm {:.4f}\n".format(mu, sd))
    log_t.close()
