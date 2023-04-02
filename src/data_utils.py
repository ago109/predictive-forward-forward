"""
Data functions and utilities.

Code for paper "The Predictive Forward-Forward Algorithm" (Ororbia & Mali, 2022)
"""
import random
import numpy as np
import io
import sys
import math

seed = 69
# set your random seed - very important to reproduce any Machine/Deep learning work
np.random.seed(seed)

class DataLoader(object):
    """
        A data loader object, meant to allow sampling w/o replacement of one or
        more named design matrices. Note that this object is iterable (and
        implements an __iter__() method).

        Args:
            design_matrices:  list of named data design matrices - [("name", matrix), ...]

            batch_size:  number of samples to place inside a mini-batch

            disable_shuffle:  if True, turns off sample shuffling (thus no sampling w/o replacement)

            ensure_equal_batches: if True, ensures sampled batches are equal in size (Default = True).
                Note that this means the very last batch, if it's not the same size as the rest, will
                reuse random samples from previously seen batches (yielding a batch with a mix of
                vectors sampled with and without replacement).
    """
    def __init__(self, design_matrices, batch_size, disable_shuffle=False,
                 ensure_equal_batches=True):
        self.batch_size = batch_size
        self.ensure_equal_batches = ensure_equal_batches
        self.disable_shuffle = disable_shuffle
        self.design_matrices = design_matrices
        if len(design_matrices) < 1:
            print(" ERROR: design_matrices must contain at least one design matrix!")
            sys.exit(1)
        self.data_len = len( self.design_matrices[0][1] )
        self.ptrs = np.arange(0, self.data_len, 1)
        if self.data_len < self.batch_size:
            print("ERROR: batch size {} is > total number data samples {}".format(
                  self.batch_size, self.data_len))
            sys.exit(1)

    def __iter__(self):
        """
            Yields a mini-batch of the form:  [("name", batch),("name",batch),...]
        """
        if self.disable_shuffle == False:
            self.ptrs = np.random.permutation(self.data_len)
        idx = 0
        while idx < len(self.ptrs): # go through each sample via the sampling pointer
            e_idx = idx + self.batch_size
            if e_idx > len(self.ptrs): # prevents reaching beyond length of dataset
                e_idx = len(self.ptrs)
            # extract sampling integer pointers
            indices = self.ptrs[idx:e_idx]
            if self.ensure_equal_batches == True:
                if indices.shape[0] < self.batch_size:
                    diff = self.batch_size - indices.shape[0]
                    indices = np.concatenate((indices, self.ptrs[0:diff]))
            # create the actual pattern vector batch block matrices
            data_batch = []
            for dname, dmatrix in self.design_matrices:
                x_batch = dmatrix[indices]
                data_batch.append((dname, x_batch))
            yield data_batch
            idx = e_idx
