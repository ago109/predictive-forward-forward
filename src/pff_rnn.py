"""
Code for paper "The Predictive Forward-Forward Algorithm" (Ororbia & Mali, 2022)

This file contains model constructor and its credit assignment code.
"""

import os
import sys
import copy
import pickle
#import dill as pickle
import tensorflow as tf
import numpy as np

### generic routines/functions

def serialize(fname, object): ## object "saving" routine
    fd = open(fname, 'wb')
    pickle.dump(object, fd)
    fd.close()

def deserialize(fname): ## object "loading" routine
    fd = open(fname, 'rb')
    object = pickle.load( fd )
    fd.close()
    return object

@tf.function
def create_competiion_matrix(z_dim, n_group, beta_scale=1.0, alpha_scale=1.0):
    """
    Competition matrix initialization function, adapted from
    (Ororbia & Kifer 2022; Nature Communications).
    """
    diag = tf.eye(z_dim)
    V_l = None
    g_shift = 0
    while (z_dim - (n_group + g_shift)) >= 0:
        if g_shift > 0:
            left = tf.zeros([1,g_shift])
            middle = tf.ones([1,n_group])
            right = tf.zeros([1,z_dim - (n_group + g_shift)])
            slice = tf.concat([left,middle,right],axis=1)
            for n in range(n_group):
                V_l = tf.concat([V_l,slice],axis=0)
        else:
            middle = tf.ones([1,n_group])
            right = tf.zeros([1,z_dim - n_group])
            slice = tf.concat([middle,right],axis=1)
            for n in range(n_group):
                if V_l is not None:
                    V_l = tf.concat([V_l,slice],axis=0)
                else:
                    V_l = slice
        g_shift += n_group
    V_l = V_l * (1.0 - diag) * beta_scale + diag * alpha_scale
    return V_l

@tf.function
def softmax(x, tau=0.0): ## temperature-controlled softmax activation
    if tau > 0.0:
        x = x / tau
    max_x = tf.expand_dims( tf.reduce_max(x, axis=1), axis=1)
    exp_x = tf.exp(tf.subtract(x, max_x))
    return exp_x / tf.expand_dims( tf.reduce_sum(exp_x, axis=1), axis=1)

@tf.custom_gradient
def _relu(x): ## FF/PFF relu variant activation
    ## modified relu
    out = tf.nn.relu(x)
    def grad(upstream):
        #dx = tf.cast(tf.math.greater_equal(x, 0.0), dtype=tf.float32) # d_relu/d_x
        dx = tf.ones(x.shape) # pretend like derivatives exist for zero values
        return upstream * dx
    return out, grad
@tf.function
def clip_fx(x): ## hard-clip activation
    return tf.clip_by_value(x, 0.0, 1.0)

### begin constructor definition

class PFF_RNN:
    """
    Basic implementation of the predictive forward-forward (FF) algorithm for a recurrent
    neural network from (Ororbia & Mali 2022).

    A "model" in this framework is defined as a pair containing an arguments
    dictionary and a theta construct, i.e., (args, theta).
    """
    def __init__(self, args=None, model_dir=None):
        theta_r = None
        theta_g = None
        if model_dir is not None:
            args = deserialize("{}config.args".format(model_dir))
            theta_r = deserialize("{}rep_params.theta".format(model_dir))
            theta_g = deserialize("{}gen_params.theta".format(model_dir))
        if args is None:
            print("ERROR: no model arguments provided...")
            sys.exit(1)
        self.args = args
        ## collect hyper-parameters
        self.seed = 69
        if args.get("seed") is not None:
            self.seed = args["seed"]
        self.x_dim = args["x_dim"]
        self.y_dim = args["y_dim"]
        self.n_units = 2000
        if args.get("n_units") is not None:
            self.n_units = args["n_units"]

        self.g_units = 20 # number of top-most latent variables for generative circuit
        if args.get("g_units") is not None:
            self.g_units = args["g_units"]
        self.K = 10
        if args.get("K") is not None:
            self.K = args["K"]
        self.beta = 0.025 #0.05 #0.1
        if args.get("beta") is not None:
            self.beta = args["beta"]

        self.gen_gamma = 1.0
        self.rec_gamma = 1.0
        self.thr = 3.0 # goodness threshold
        if args.get("thr") is not None:
            self.thr = args["thr"]
        self.alpha = 0.3 # dampening factor
        if args.get("alpha") is not None:
            self.alpha = args["alpha"]

        self.y_scale = 5.0 # 1.0
        # stats for peer normalization (if used)
        self.eps_r = 0.01 # noise factor for representation circuit
        if args.get("eps_r") is not None:
            self.eps_r = args["eps_r"]
        self.eps_g = 0.025 # noise factor for generative circuit
        if args.get("eps_g") is not None:
            self.eps_g = args["eps_g"]

        ## Set up parameter construct - theta_r and theta_g
        if theta_r is None: ## representation circuit params
            initializer = tf.compat.v1.keras.initializers.Orthogonal()
            self.b1 = tf.Variable(initializer([1, self.n_units]))
            self.b2 = tf.Variable(initializer([1, self.n_units]))
            self.W1 = tf.Variable(initializer([self.x_dim, self.n_units]))
            self.V2 = tf.Variable(initializer([self.n_units, self.n_units])) # inner feedback
            self.W2 = tf.Variable(initializer([self.n_units, self.n_units]))
            self.V = tf.Variable(initializer([self.y_dim, self.n_units])) # top to inner feedback
            self.W = tf.Variable(initializer([self.n_units, self.y_dim])) # softmax output
            self.b = tf.Variable(tf.zeros([1, self.y_dim]))
            initializer = tf.compat.v1.keras.initializers.RandomUniform(minval=0.0, maxval=0.05)

            self.M1 = create_competiion_matrix(self.n_units, n_group=10)
            self.M2 = create_competiion_matrix(self.n_units, n_group=10)
            self.L1 = tf.Variable(initializer([self.n_units, self.n_units])) # lateral lyr 1
            self.L2 = tf.Variable(initializer([self.n_units, self.n_units])) # lateral lyr 2
            theta_r = [self.L1,self.L2,self.b1,self.b2,self.W1,self.W2,self.V2,self.V,self.W,self.b] ## theta_r
        else:
            self.M1 = create_competiion_matrix(self.n_units, n_group=10)
            self.M2 = create_competiion_matrix(self.n_units, n_group=10)
            self.L1 = theta_r[0]
            self.L2 = theta_r[1]
            self.b1 = theta_r[2]
            self.b2 = theta_r[3]
            self.W1 = theta_r[4]
            self.W2 = theta_r[5]
            self.V2 = theta_r[6]
            self.V = theta_r[7]
            self.W = theta_r[8]
            self.b = theta_r[9]
        self.theta = theta_r

        if theta_g is None: ## generative circuit params
            self.Gy = tf.Variable(initializer([self.g_units, self.n_units]))
            self.G2 = tf.Variable(initializer([self.n_units, self.n_units]))
            self.G1 = tf.Variable(initializer([self.n_units, self.x_dim]))
            theta_g = [self.Gy,self.G1,self.G2] ## theta_g
        else:
            self.Gy = theta_g[0]
            self.G1 = theta_g[1]
            self.G2 = theta_g[2]
        self.theta_g = theta_g

        ## activation functions and latent states/stat variables
        self.z_g = None # top-most generative latent state
        self.fx = _relu ## internal activation function
        self.ofx = clip_fx ## predictive activation output function
        self.gfx = _relu ## generative activation output function
        self.z1 = None
        self.z0_hat = None

    def save_model(self, model_dir):
        """
        Save current model configuration and synaptic parameters (of both
        representation & generative circuits) to disk.

        Args:
            model_dir: directory to save model config
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        serialize("{}config.args".format(model_dir), self.args)
        serialize("{}rep_params.theta".format(model_dir), self.theta)
        serialize("{}gen_params.theta".format(model_dir), self.theta_g)

    def calc_goodness(self, z, thr):
        """
        Calculates the "goodness" of an activation vector.

        Args:
            z: activation vector/matrix
            thr: goodness threshold

        Returns:
            goodness scalar of z
        """
        z_sqr = tf.math.square(z)
        delta = tf.reduce_sum(z_sqr, axis=1, keepdims=True)
        #delta = delta - thr #  maximize for positive samps, minimize for negative samps
        delta = -delta + thr #  minimize for positive samps, maximize for negative samps
        # gets the probability P(pos)
        p = tf.nn.sigmoid(delta)
        eps = 1e-5
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        return p, delta

    def calc_loss(self, z, lab, thr, keep_batch=False):
        """
        Calculates the local loss of an activation vector.

        Args:
            z: activation vector/matrix (vector/matrix)
            lab: data "type" binary label (1 for pos, 0 for neg) (vector/matrix)
            thr: goodness threshold

        Returns:
            goodness scalar of z
        """
        p, logit = self.calc_goodness(z, thr)
        ## the loss below is what the original PFF paper used & adheres to Eqn 3
        CE = tf.math.maximum(logit, 0) - logit * lab + tf.math.log(1. + tf.math.exp(-tf.math.abs(logit)))
        ## the commented-out loss below, however, also works just fine
        #CE = tf.nn.softplus(-logit) * lab + tf.nn.softplus(logit) * (1.0 - lab)
        L = tf.reduce_sum(CE, axis=1, keepdims=True)
        if keep_batch == True:
            return L
        L = tf.reduce_mean(L)
        return L

    def forward(self, x):
        """
        Forward propagates x thru rep circuit

        Args:
            x: sensory input (vector/matrix)

        Returns:
            list of layer-wise activities
        """
        z1 = self.fx(tf.matmul(self.normalize(x), self.W1) + self.b1)
        z2 = self.fx(tf.matmul(self.normalize(z1), self.W2) + self.b2)
        z3 = softmax(tf.matmul(self.normalize(z2), self.W) + self.b)
        return [z1,z2,z3] # return all latents

    def classify(self, x):
        """
        Categorizes sensory input x

        Args:
            x: sensory input (vector/matrix)

        Returns:
            y_hat, probability distribution over labels (vector/matrix)
        """
        z = self.forward(x)
        y_hat = z[len(z)-1]
        return y_hat

    def infer(self, x, y, lab, z_lat, K, opt=None, g_opt=None, reg_lambda=0.0,
              zero_y=False, g_reg_lambda=0.0):
        """
        Simulates the PFF intertwined inference-and-learning process for
        the underlying dual-circuit system.

        Args:
            x: sensory input (vector/matrix)
            y: sensory class label (vector/matrix)
            lab: data "type" binary label (1 for pos, 0 for neg) (vector/matrix)
            z_lat: list of initial conditions for model's representation activities
                   (usually provided by an initial forward pass with .forward(x) )
            K: number of simulation steps
            opt: rep circuit optimizer
            g_opt: gen circuit optimizer
            reg_lambda: regularization coefficient (for representation synapses)
            zero_y: "zero out" the y-vector top-down context
            g_reg_lambda: regularization coefficient (for generative synapses)

        Returns:
            (global energy value (goodness + regularization), label distribution matrix,
             generative loss, x reconstruction)
        """
        if self.rec_gamma > 0.0:
            self.theta = [self.L1,self.L2,self.b1,self.b2,self.W1,self.W2,self.V2,self.V,self.W,self.b]
        else:
            self.theta = [self.b1,self.b2,self.W1,self.W2,self.V2,self.V,self.W,self.b]

        calc_grad = False
        if opt is not None:
            calc_grad = True
        # update generative model
        self.z_g = tf.Variable(tf.zeros([x.shape[0], self.g_units]))
        for k in range(K):
            # update representation model
            z_lat, L, delta = self.step(x,y,lab,z_lat,calc_grad=calc_grad,
                                        zero_y=zero_y, reg_lambda=reg_lambda)
            if opt is not None: ## update synapses
                bound = 1.0 #5.0 # 1.0
                for l in range(len(delta)):
                    delta[l] = tf.clip_by_value(delta[l], -bound, bound) # clip update by projection
                opt.apply_gradients(zip(delta, self.theta))
            if self.gen_gamma > 0.0: ## update generative model
                grad_f = True #False
                Lg, x_hat = self.update_generator(x,y,z_lat,g_opt,reg_lambda=g_reg_lambda,grad_f=grad_f)
            else:
                Lg = 0.0
                x_hat = x * 0

        y_hat = z_lat[len(z_lat)-1]
        return L, y_hat, Lg, x_hat

    def sample(self, n_s=0, z=None, y=None): # samples generative circuit
        """
        Samples the generative circuit within this current neural system.

        Args:
            n_s: number of samples to synthesize/confabulate
            z: top-most externally produced input sample (from a prior); (vector/matrix)
            y: sensory class label (vector/matrix)

        Returns:
            samples of the bottom-most sensory layer
        """
        if z is None:
            eps_sigma = 0.05
            #eps = tf.random.normal([y.shape[0],self.n_units], 0.0, eps_sigma) #* 0
            z_in = self.normalize(self.gfx(y))
            z2 = self.gfx(tf.matmul(z_in,self.Gy))# + self.c)
        else:
            z2 = self.gfx(z)
        #z2 = self.gfx(z2)
        z1 = self.gfx(tf.matmul(self.normalize(z2),self.G2))# + self.c2)
        #z0 = tf.matmul(self.normalize(z1),self.G1) #
        z0 = self.ofx(tf.matmul(self.normalize(z1),self.G1))# + self.c1)
        return z0

    def update_generator(self, x, y, z_lat, opt, reg_lambda=0.0001, grad_f=True):
        '''
        Internal routine for adjusting synapses of generative circuit
        '''
        z0 = x
        z1 = z_lat[0]
        z2 = z_lat[1]
        eps_sigma = self.eps_g #0.025 #0.055 # 0.05 #0.02 #0.1
        eps1 = tf.random.normal([x.shape[0],self.n_units], 0.0, eps_sigma) #* 0
        eps2 = tf.random.normal([x.shape[0],self.n_units], 0.0, eps_sigma) #* 0
        with tf.GradientTape(persistent=True) as tape:
            z3_bar = self.gfx(self.z_g)
            z2_hat = tf.matmul(self.normalize(z3_bar),self.Gy) #+ self.c
            #z2_hat = self.fx(tf.matmul(self.z_g,self.Gy))
            z2_bar = self.gfx(z2_hat) #z2 #self.fx(z2 + eps2)
            z1_hat = tf.matmul(tf.stop_gradient(self.normalize(self.gfx(z2 + eps2))),self.G2) #+ self.c2
            #z1_hat = self.fx(tf.matmul(z2_bar,self.G2))
            z1_bar = self.gfx(z1_hat) #z1 #self.fx(z1 + eps1)
            #z0_hat = tf.matmul(z1_bar,self.G1) #
            z0_hat = self.ofx(tf.matmul(tf.stop_gradient(self.normalize(self.gfx(z1 + eps1))),self.G1)) #+ self.c1

            e2 = z2_bar - z2#_bar
            L2 = tf.reduce_mean(tf.reduce_sum(tf.math.square(e2),axis=1,keepdims=True))
            e1 = z1_bar - z1#_bar
            L1 = tf.reduce_mean(tf.reduce_sum(tf.math.square(e1),axis=1,keepdims=True))
            e0 = z0_hat - z0
            L0 = tf.reduce_mean(tf.reduce_sum(tf.math.square(e0),axis=1,keepdims=True))

            reg = 0.0
            if reg_lambda > 0.0: # weight decay
                reg = (tf.norm(self.G1) + tf.norm(self.G2) + tf.norm(self.Gy)) * reg_lambda
            L = L2 + L1 + L0 + reg
        Lg = L2 + L1 + L0
        if grad_f == True:
            delta = tape.gradient(L, self.theta_g)
            bound = 1.0 #5.0 # 1.0
            for l in range(len(delta)):
                delta[l] = tf.clip_by_value(delta[l], -bound, bound) # clip update by projection
            opt.apply_gradients(zip(delta, self.theta_g))
            # for l in range(len(self.theta_g)):
            #     self.theta_g[l].assign(tf.clip_by_norm(self.theta_g[l], 5.0, axes=1))

        d_z = tape.gradient(L2, self.z_g)
        self.z_g.assign(self.z_g - d_z * self.beta)

        return Lg, z0_hat # return generative loss

    def step(self, x, y, lab, z_lat, calc_grad=False, zero_y=False, reg_lambda=0.0):
        '''
        Internal full simulation step routine (inference and credit assignment)
        '''
        y_ = y * self.y_scale
        Npos = tf.reduce_sum(lab)

        eps_sigma = self.eps_r # 0.01 #0.05 # kmnist
        #eps_sigma = 0.025 # mnist
        eps1 = tf.random.normal([x.shape[0],self.n_units], 0.0, 1.0) * eps_sigma
        eps2 = tf.random.normal([x.shape[0],self.n_units], 0.0, 1.0) * eps_sigma

        with tf.GradientTape() as tape:
            z1_tm1 = tf.stop_gradient(z_lat[0])
            z2_tm1 = tf.stop_gradient(z_lat[1])

            z1 = tf.matmul(self.normalize(x),self.W1) + tf.matmul(self.normalize(z2_tm1),self.V2) + self.b1 + eps1
            if self.rec_gamma > 0.0:
                L1 = tf.nn.relu(self.L1)
                L1 = L1 * self.M1 * (1. - tf.eye(self.L1.shape[0])) - L1 * tf.eye(self.L1.shape[0])
                z1 = z1 - tf.matmul(z1_tm1, L1) * self.rec_gamma
            z1 = self.fx(z1) * (1.0 - self.alpha) + z1_tm1 * self.alpha

            z2 = tf.matmul(self.normalize(z1_tm1),self.W2) + tf.matmul(y_,self.V) + self.b2 + eps2
            if self.rec_gamma > 0.0:
                L2 = tf.nn.relu(self.L2)
                L2 = L2 * self.M2 * (1. - tf.eye(self.L2.shape[0])) - L2 * tf.eye(self.L2.shape[0])
                z2 = z2 - tf.matmul(z2_tm1, L2) * self.rec_gamma
            z2 = self.fx(z2) * (1.0 - self.alpha) + z2_tm1 * self.alpha

            z3 = softmax(tf.matmul(self.normalize(z2_tm1),self.W) + self.b) #tf.nn.softmax(tf.matmul(self.normalize(z2),self.W) + self.b)

            ## calc loss
            L1 = self.calc_loss(z1, lab, thr=self.thr)
            L2 = self.calc_loss(z2, lab, thr=self.thr)
            if zero_y == True:
                L3 = tf.reduce_sum(-tf.reduce_sum(y_ * tf.math.log(z3), axis=1, keepdims=True)) * 0
            else:
                L3 = tf.reduce_sum(-tf.reduce_sum(y_ * tf.math.log(z3) * lab, axis=1, keepdims=True) * lab)
                L3 = L3 / Npos
            reg = 0.0
            if reg_lambda > 0.0: # weight decay
                reg = (tf.norm(self.W1) + tf.norm(self.W2) + tf.norm(self.V2) +
                       tf.norm(self.V) + tf.norm(self.W)) * reg_lambda
            L = L3 + L2 + L1 + reg
        Lg = L2 + L1
        delta = None
        if calc_grad == True:
            delta = tape.gradient(L, self.theta)
        return [z1,z2,z3], Lg, delta

    def _step(self, x, y, z_lat, thr=None):
        '''
        Internal simulation step, without credit assignment
        '''
        thr_ = thr
        y_ = y * self.y_scale

        z1_tm1 = z_lat[0]
        z2_tm1 = z_lat[1]

        z1 = tf.matmul(self.normalize(x),self.W1) + tf.matmul(self.normalize(z2_tm1),self.V2) + self.b1
        if self.rec_gamma > 0.0:
            L1 = tf.nn.relu(self.L1)
            L1 = L1 * self.M1 * (1. - tf.eye(self.L1.shape[0])) - L1 * tf.eye(self.L1.shape[0])
            z1 = z1 - tf.matmul(z1_tm1, L1) * self.rec_gamma
        z1 = self.fx(z1) * (1.0 - self.alpha) + z1_tm1 * self.alpha

        z2 = tf.matmul(self.normalize(z1_tm1),self.W2) + tf.matmul(y_,self.V) + self.b2
        if self.rec_gamma > 0.0:
            L2 = tf.nn.relu(self.L2)
            L2 = L2 * self.M2 * (1. - tf.eye(self.L2.shape[0])) - L2 * tf.eye(self.L2.shape[0])
            z2 = z2 - tf.matmul(z2_tm1, L2) * self.rec_gamma
        z2 = self.fx(z2) * (1.0 - self.alpha) + z2_tm1 * self.alpha
        ## calc goodness values
        if thr_ is None:
            thr_ = self.thr
        p1, logit1 = self.calc_goodness(z1, thr_)
        p2, logit2 = self.calc_goodness(z2, thr_)

        return [z1,z2], [logit1,logit2]

    def get_latent(self, x, y, K, use_y_hat=False):
        self.z1 = None
        self.z0_hat = 0.0
        z_lat = self.forward(x)
        y_hat = z_lat[len(z_lat)-1]
        self.z_g = tf.Variable(tf.zeros([x.shape[0], self.g_units]))
        y_ = y
        if use_y_hat == True:
            y_ = y_hat
        for k in range(K):
            self._infer_latent(x,y_,z_lat)
        #self.z0_hat = self.ofx(tf.matmul(self.normalize(self.gfx(self.z1)),self.G1))
        #self.z0_hat = self.z0_hat/(K * 1.0)
        return self.z_g + 0

    def _infer_latent(self, x, y, z_lat):
        '''
        Internal routine for inferring the generative circuit's top-most latent state activity
        '''
        y_ = y * self.y_scale

        z1_tm1 = z_lat[0]
        z2_tm1 = z_lat[1]
        with tf.GradientTape() as tape:
            z1 = tf.matmul(self.normalize(x),self.W1) + tf.matmul(self.normalize(z2_tm1),self.V2) + self.b1
            if self.rec_gamma > 0.0:
                L1 = tf.nn.relu(self.L1)
                L1 = L1 * self.M1 * (1. - tf.eye(self.L1.shape[0])) - L1 * tf.eye(self.L1.shape[0])
                z1 = z1 - tf.matmul(z1_tm1, L1) * self.rec_gamma
            z1 = self.fx(z1) * (1.0 - self.alpha) + z1_tm1 * self.alpha
            self.z1 = z1 + 0

            z2 = tf.matmul(self.normalize(z1_tm1),self.W2) + tf.matmul(y_,self.V) + self.b2
            if self.rec_gamma > 0.0:
                L2 = tf.nn.relu(self.L2)
                L2 = L2 * self.M2 * (1. - tf.eye(self.L2.shape[0])) - L2 * tf.eye(self.L2.shape[0])
                z2 = z2 - tf.matmul(z2_tm1, L2) * self.rec_gamma
            z2 = self.fx(z2) * (1.0 - self.alpha) + z2_tm1 * self.alpha

            z3_bar = self.gfx(self.z_g)
            z2_hat = tf.matmul(self.normalize(z3_bar),self.Gy)
            z2_bar = self.gfx(z2_hat)
            e2 = z2_bar - z2#_bar
            L2 = tf.reduce_mean(tf.reduce_sum(tf.math.square(e2),axis=1,keepdims=True))

        d_z = tape.gradient(L2, self.z_g)
        self.z_g.assign(self.z_g - d_z * self.beta)

        z0_hat = self.ofx(tf.matmul(self.normalize(self.gfx(z1)),self.G1))
        self.z0_hat = z0_hat

    def normalize(self, z_state, a_scale=1.0): ## norm
        '''
        Internal routine for normalizing state vector/matrix "z_state"
        '''
        eps = 1e-8
        L2 = tf.norm(z_state, ord=2, axis=1, keepdims=True)
        z_state = z_state / (L2 + eps)
        return z_state * a_scale
