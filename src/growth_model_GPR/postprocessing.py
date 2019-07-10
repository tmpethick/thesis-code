#======================================================================
#
#     This module contains routines to postprocess the VFI 
#     solutions.
#
#     Simon Scheidegger, 01/19
#======================================================================

import os
import pickle
import numpy as np
from src.models import DKLGPModel

class PostProcessing(object):
    def __init__(self, params):
        self.params = params

    def ls_error(self):
        with open(self.params.error_file, 'w') as file:

            np.random.seed(0)
            unif=np.random.rand(self.params.No_samples_postprocess, self.params.n_agents)
            k_sample=self.params.k_bar+(unif)*(self.params.k_up-self.params.k_bar)
            to_print=np.empty((1,3))

            for i in range(self.params.numstart, self.params.numits-1):
                sum_diffs = 0
                diff = 0

                model_old = DKLGPModel.load(self.params.model_dir + str(i))
                model_new = DKLGPModel.load(self.params.model_dir + str(i+1))
                # with open(os.path.join(self.params.model_dir + str(i), 'sklearn.pickle'), 'rb') as fd:
                #     model_old = pickle.load(fd)
                # with open(os.path.join(self.params.model_dir + str(i+1), 'sklearn.pickle'), 'rb') as fd:
                #     model_new = pickle.load(fd)

                y_pred_old, sigma_old = model_old.get_statistics(k_sample, full_cov=False)
                y_pred_new, sigma_new = model_new.get_statistics(k_sample, full_cov=False)
                # y_pred_old, sigma_old = model_old.predict(k_sample, return_std=True)
                # y_pred_new, sigma_new = model_new.predict(k_sample, return_std=True)

                # plot predictive mean and 95% quantiles
                #for j in range(num_points):
                    #print k_sample[j], " ",y_pred_new[j], " ",y_pred_new[j] + 1.96*sigma_new[j]," ",y_pred_new[j] - 1.96*sigma_new[j]

                diff = y_pred_old-y_pred_new
                max_abs_diff=np.amax(np.fabs(diff))
                average = np.average(np.fabs(diff))

                to_print[0,0]= i+1
                to_print[0,1]= max_abs_diff
                to_print[0,2]= average

                np.savetxt(file, to_print, fmt='%2.16f')
