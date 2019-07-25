import numpy as np


#====================================================================== 
#utility function u(c,l) 

class Econ(object):
    def __init__(self, params):
        self.params = params

    def utility(self, cons=[], lab=[]):
        sum_util=0.0
        n=len(cons)
        for i in range(n):
            nom1=(cons[i]/self.params.big_A)**(1.0-self.params.gamma) -1.0
            den1=1.0-self.params.gamma

            nom2=(1.0-self.params.psi)*((lab[i]**(1.0+self.params.eta)) -1.0)
            den2=1.0+self.params.eta

            sum_util+=(nom1/den1 - nom2/den2)

        util=sum_util

        return util


    #======================================================================
    # output_f

    def output_f(self, kap=[], lab=[]):
        fun_val = self.params.big_A*(kap**self.params.psi)*(lab**(1.0 - self.params.psi))
        return fun_val

    #======================================================================

    # transformation to comp domain -- range of [k_bar, k_up]

    def box_to_cube(self, knext=[]):
        n=len(knext)
        knext_box = knext[0:n]
        knext_dummy = knext[0:n]

        scaling_dept = (self.params.range_cube/(self.params.k_up  - self.params.k_bar))   #scaling for kap

        #transformation onto cube [0,1]^d
        for i in range(n):
            #prevent values outside the box
            if  knext[i] > self.params.k_up:
                knext_dummy[i] = self.params.k_up
            elif knext[i] < self.params.k_bar:
               knext_dummy[i] = self.params.k_bar
            else:
                knext_dummy[i] = knext[i]
            #transformation to sparse grid domain
            knext_box[i] = (knext_dummy[i] - self.params.k_bar)*scaling_dept

        return knext_box

#======================================================================  
