#======================================================================
#
#     This routine interfaces with Gaussian Process Regression
#     The crucial part is 
#
#     y[iI] = solver.initial(Xtraining[iI], n_agents)[0]  
#     => at every training point, we solve an optimization problem
#
#     Simon Scheidegger, 01/19
#======================================================================
import os 
import numpy as np
from .nonlinear_solver import NonlinearSolver
import pickle as pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

#======================================================================

class Interpolation(object):
    def __init__(self, params, nonlinear_solver):
        self.params = params
        self.nonlinear_solver = nonlinear_solver

    def GPR_init(self, iteration):

        print("hello from step ", iteration)


        #fix seed
        np.random.seed(666)

        #generate sample aPoints
        dim = self.params.n_agents
        Xtraining = np.random.uniform(self.params.k_bar, self.params.k_up, (self.params.No_samples, dim))
        y = np.zeros(self.params.No_samples, float) # training targets

        # solve bellman equations at training points
        for iI in range(len(Xtraining)):
            y[iI] = self.nonlinear_solver.initial(Xtraining[iI], self.params.n_agents)[0]

        #for iI in range(len(Xtraining)):
            #print Xtraining[iI], y[iI]

        # Instantiate a Gaussian Process model
        kernel = RBF()

        #kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        #kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)

        #kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2)) \
        #+ WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e+0))

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(Xtraining, y)


        #save the model to a file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        output_file = os.path.join(dir_path, self.params.filename + str(iteration) + ".pcl")
        print(output_file)
        with open(output_file, 'wb') as fd:
            pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
            print("data of step ", iteration ,"  written to disk")
            print(" -------------------------------------------")
        fd.close()


    def GPR_iter(self, iteration):
        # Load the model from the previous iteration step
        dir_path = os.path.dirname(os.path.realpath(__file__))
        restart_data = os.path.join(dir_path, self.params.filename + str(iteration - 1) + ".pcl")
        with open(restart_data, 'rb') as fd_old:
            gp_old = pickle.load(fd_old)
            print("data from iteration step ", iteration - 1, "loaded from disk")
        fd_old.close()

        ##generate sample aPoints
        np.random.seed(666)  # fix seed
        dim = self.params.n_agents
        Xtraining = np.random.uniform(self.params.k_bar, self.params.k_up, (self.params.No_samples, dim))
        y = np.zeros(self.params.No_samples, float)  # training targets

        # solve bellman equations at training points
        for iI in range(len(Xtraining)):
            y[iI] = self.nonlinear_solver.iterate(Xtraining[iI], self.params.n_agents, gp_old)[0]

            # print data for debugging purposes
        # for iI in range(len(Xtraining)):
        # print Xtraining[iI], y[iI]

        # Instantiate a Gaussian Process model
        kernel = RBF()

        # Instantiate a Gaussian Process model
        # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2)) \
        # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e+0))

        # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 2e2))
        # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(Xtraining, y)

        ##save the model to a file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        output_file = os.path.join(dir_path, self.params.filename + str(iteration) + ".pcl")
        print(output_file)
        with open(output_file, 'wb') as fd:
            pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
            print("data of step ", iteration, "  written to disk")
            print(" -------------------------------------------")
        fd.close()

#======================================================================

