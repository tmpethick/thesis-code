#======================================================================
# 
#     sets the parameters for the model
#     "Growth Model"
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     Simon Scheidegger, 01/19
#====================================================================== 

import os
import numpy as np

#====================================================================== 

class Parameters(object):
    #======================================================================

    # Number of test points to compute the error in the postprocessing

    def __init__(self, 
        n_agents = 2,
        numstart = 1,
        numits = 7,
        output_dir = "output/",
        model_dir = "restart_file_step_",
        error_file = "errors.txt",
        beta = 0.96,
        zeta = 0.5,
        psi = 0.36,
        gamma = 2.0,
        delta = 0.06,
        eta = 1,
        k_bar = 0.2,
        k_up = 3.0,
        c_bar = 1e-2,
        c_up = 10.0,
        l_bar = 1e-2,
        l_up = 10.0,
        inv_bar = 1e-2,
        inv_up = 10.0,
        No_samples = None,
        No_samples_postprocess = 20,
    ):
        # How many training points for GPR
        self.n_agents = n_agents # number of continuous dimensions of the model
        if No_samples is None:
            self.No_samples = 10*n_agents
        else:
            self.No_samples = No_samples

        # control of iterations
        self.numstart = numstart
        self.numits = numits

        #directory = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(output_dir, model_dir)
        self.error_file = os.path.join(output_dir, error_file)

        #======================================================================

        # Model Paramters
        self.beta = beta
        self.zeta = zeta
        self.psi = psi
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.big_A = (1.0-beta)/(psi*beta)

        # Ranges For States
        self.k_bar = k_bar
        self.k_up = k_up

        # Ranges for Controls
        self.c_bar = c_bar
        self.c_up = c_up

        self.l_bar = l_bar
        self.l_up = l_up

        self.inv_bar = inv_bar
        self.inv_up = inv_up
        self.range_cube = k_up - k_bar # range of [0..1]^d in 1D

        self.No_samples_postprocess = No_samples_postprocess

