#======================================================================
#
#     This routine solves an infinite horizon growth model 
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - scikit-learn GPR (https://scikit-learn.org)
#
#     Simon Scheidegger, 01/19 
#======================================================================
from src.growth_model_GPR.econ import Econ
from src.growth_model_GPR.ipopt_wrapper import IPOptWrapper
from .parameters import Parameters
from .nonlinear_solver import NonlinearSolver   #solves opt. problems for terminal VF
from .interpolation import  Interpolation       #interface to sparse grid library/terminal VF
from .postprocessing import PostProcessing      #computes the L2 and Linfinity error of the model
import numpy as np


#======================================================================
# Start with Value Function Iteration



class GrowthModel(object):
    def __init__(self, **kwargs):
        self.params = Parameters(**kwargs)

        self.econ = Econ(self.params)
        self.ipopt = IPOptWrapper(self.params, self.econ)
        self.nonlinear_solver = NonlinearSolver(self.params, self.ipopt)

        self.interpolation = Interpolation(self.params, self.nonlinear_solver)
        self.post = PostProcessing(self.params)

    def loop(self):
        for i in range(self.params.numstart, self.params.numits):
        # terminal value function
            if (i==1):
                print("start with Value Function Iteration")
                self.interpolation.GPR_init(i)

            else:
                print("Now, we are in Value Function Iteration step", i)
                self.interpolation.GPR_iter(i)


        #======================================================================
        print("===============================================================")
        print(" ")
        print(" Computation of a growth model of dimension ", self.params.n_agents ," finished after ", self.params.numits, " steps")
        print(" ")
        print("===============================================================")
        #======================================================================

        # compute errors
        avg_err=self.post.ls_error(self.params.n_agents, self.params.numstart, self.params.numits, self.params.No_samples_postprocess)

        #======================================================================
        print("===============================================================")
        print(" ")
        #print " Errors are computed -- see error.txt"
        print(" ")
        print("===============================================================")
        #======================================================================
