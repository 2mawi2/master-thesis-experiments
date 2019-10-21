import os
import pickle

import numpy as np

from src.common.utils import *

preprocessor_path = os.path.join(get_models_dir(algorithm="policy"), "scaler",
                                 "scaler.bin")  # "./models/scaler/scaler.bin"


class Preprocessor:
    """
        handles scaling and centring of environment data
     """

    def __init__(self, obs_dim, load=False):
        self.eval = load

        if load:  # load saved values form disk
            print("loading preprocessor...")
            with open(preprocessor_path, "rb") as f:
                self.vars, self.means, self.m, self.n, self.first_time = pickle.load(f)

        else:  # default values
            self.vars = np.zeros(obs_dim)
            self.means = np.zeros(obs_dim)
            self.m = 0
            self.n = 0
            self.first_time = True

    def update(self, x):
        if self.eval:
            return

        if self.first_time:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_time = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))  # center and scale on observed max min
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self):
        scale, offset = 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means
        return scale, offset

    def close(self):
        print("saving preprocessor...")
        with open(preprocessor_path, "wb+") as f:
            pickle.dump((self.vars, self.means, self.m, self.n, self.first_time), f)
