#=======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT 
#
#     Simon Scheidegger, 06/17
#
#=======================================================================

from .econ import Econ
import numpy as np

#=======================================================================
#   Objective Function to start VFI (in our case, the value function)

class IPOptWrapper(object):
    def __init__(self, params, econ):
        self.econ = econ
        self.params = params

    def EV_F(self, X, k_init, n_agents):

        # Extract Variables
        cons=X[0:n_agents]
        lab=X[n_agents:2*n_agents]
        inv=X[2*n_agents:3*n_agents]

        knext= (1-self.params.delta)*k_init + inv

        # Compute Value Function
        VT_sum=self.econ.utility(cons, lab) + self.params.beta*self.V_INFINITY(knext)

        return VT_sum

    # V infinity
    def V_INFINITY(self, k=[]):
        e=np.ones(len(k))
        c=self.econ.output_f(k,e)
        v_infinity=self.econ.utility(c,e)/(1-self.params.beta)
        return v_infinity

    #=======================================================================
    #   Objective Function during VFI (note - we need to interpolate on an "old" GPR)

    def EV_F_ITER(self, X, k_init, n_agents, gp_old):

        # Extract Variables
        cons=X[0:n_agents]
        lab=X[n_agents:2*n_agents]
        inv=X[2*n_agents:3*n_agents]

        knext= (1-self.params.delta)*k_init + inv

        #transform to comp. domain of the model
        knext_cube = self.econ.box_to_cube(knext)

        # initialize correct data format for training point
        s = (1,n_agents)
        Xtest = np.zeros(s)
        Xtest[0,:] = knext_cube

        # interpolate the function, and get the point-wise std.
        #V_old, sigma_test = gp_old.get_statistics(Xtest, full_cov=False)
        #V_old, sigma_test = gp_old.get_statistics(Xtest, full_cov=False)
        V_old, sigma_test = gp_old.predict(Xtest, return_std=True)

        # aggregate hyperparams
        if V_old.ndim == 3:
            V_old = np.mean(V_old, axis=0)


        VT_sum = self.econ.utility(cons, lab) + self.params.beta*V_old

        return VT_sum


    #=======================================================================
    #   Computation of gradient (first order finite difference) of initial objective function

    def EV_GRAD_F(self, X, k_init, n_agents):

        N=len(X)
        GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
        h=1e-4


        for ixN in range(N):
            xAdj=np.copy(X)

            if (xAdj[ixN] - h >= 0):
                xAdj[ixN]=X[ixN] + h
                fx2=self.EV_F(xAdj, k_init, n_agents)

                xAdj[ixN]=X[ixN] - h
                fx1=self.EV_F(xAdj, k_init, n_agents)

                GRAD[ixN]=(fx2-fx1)/(2.0*h)

            else:
                xAdj[ixN]=X[ixN] + h
                fx2=self.EV_F(xAdj, k_init, n_agents)

                xAdj[ixN]=X[ixN]
                fx1=self.EV_F(xAdj, k_init, n_agents)
                GRAD[ixN]=(fx2-fx1)/h

        return GRAD

    #=======================================================================
    #   Computation of gradient (first order finite difference) of the objective function

    def EV_GRAD_F_ITER(self, X, k_init, n_agents, gp_old):

        N=len(X)
        GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
        h=1e-4


        for ixN in range(N):
            xAdj=np.copy(X)

            if (xAdj[ixN] - h >= 0):
                xAdj[ixN]=X[ixN] + h
                fx2=self.EV_F_ITER(xAdj, k_init, n_agents, gp_old)

                xAdj[ixN]=X[ixN] - h
                fx1=self.EV_F_ITER(xAdj, k_init, n_agents, gp_old)

                GRAD[ixN]=(fx2-fx1)/(2.0*h)

            else:
                xAdj[ixN]=X[ixN] + h
                fx2=self.EV_F_ITER(xAdj, k_init, n_agents, gp_old)

                xAdj[ixN]=X[ixN]
                fx1=self.EV_F_ITER(xAdj, k_init, n_agents, gp_old)
                GRAD[ixN]=(fx2-fx1)/h

        return GRAD

    #======================================================================
    #   Equality constraints for the first time step of the model

    def EV_G(self, X, k_init, n_agents):
        N=len(X)
        M=3*n_agents+1  # number of constraints
        G=np.empty(M, float)

        # Extract Variables
        cons=X[:n_agents]
        lab=X[n_agents:2*n_agents]
        inv=X[2*n_agents:3*n_agents]


        # first n_agents equality constraints
        for i in range(n_agents):
            G[i]=cons[i]
            G[i + n_agents]=lab[i]
            G[i+2*n_agents]=inv[i]


        f_prod=self.econ.output_f(k_init, lab)
        Gamma_adjust=0.5*self.params.zeta*k_init*((inv/k_init - self.params.delta)**2.0)
        sectors_sum=cons + inv - self.params.delta*k_init - (f_prod - Gamma_adjust)
        G[3*n_agents]=np.sum(sectors_sum)

        return G

    #======================================================================
    #   Equality constraints during the VFI of the model

    def EV_G_ITER(self, X, k_init, n_agents):
        N=len(X)
        M=3*n_agents+1  # number of constraints
        G=np.empty(M, float)

        # Extract Variables
        cons=X[:n_agents]
        lab=X[n_agents:2*n_agents]
        inv=X[2*n_agents:3*n_agents]


        # first n_agents equality constraints
        for i in range(n_agents):
            G[i]=cons[i]
            G[i + n_agents]=lab[i]
            G[i+2*n_agents]=inv[i]


        f_prod=self.econ.output_f(k_init, lab)
        Gamma_adjust=0.5*self.params.zeta*k_init*((inv/k_init - self.params.delta)**2.0)
        sectors_sum=cons + inv - self.params.delta*k_init - (f_prod - Gamma_adjust)
        G[3*n_agents]=np.sum(sectors_sum)

        return G

    #======================================================================
    #   Computation (finite difference) of Jacobian of equality constraints
    #   for first time step

    def EV_JAC_G(self, X, flag, k_init, n_agents):
        N=len(X)
        M=3*n_agents+1
        NZ=M*N
        A=np.empty(NZ, float)
        ACON=np.empty(NZ, int)
        AVAR=np.empty(NZ, int)

        # Jacobian matrix structure

        if (flag):
            for ixM in range(M):
                for ixN in range(N):
                    ACON[ixN + (ixM)*N]=ixM
                    AVAR[ixN + (ixM)*N]=ixN

            return (ACON, AVAR)

        else:
            # Finite Differences
            h=1e-4
            gx1=self.EV_G(X, k_init, n_agents)

            for ixM in range(M):
                for ixN in range(N):
                    xAdj=np.copy(X)
                    xAdj[ixN]=xAdj[ixN]+h
                    gx2=self.EV_G(xAdj, k_init, n_agents)
                    A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
            return A

    #======================================================================
    #   Computation (finite difference) of Jacobian of equality constraints
    #   during iteration

    def EV_JAC_G_ITER(self, X, flag, k_init, n_agents):
        N=len(X)
        M=3*n_agents+1
        NZ=M*N
        A=np.empty(NZ, float)
        ACON=np.empty(NZ, int)
        AVAR=np.empty(NZ, int)

        # Jacobian matrix structure

        if (flag):
            for ixM in range(M):
                for ixN in range(N):
                    ACON[ixN + (ixM)*N]=ixM
                    AVAR[ixN + (ixM)*N]=ixN

            return (ACON, AVAR)

        else:
            # Finite Differences
            h=1e-4
            gx1=self.EV_G_ITER(X, k_init, n_agents)

            for ixM in range(M):
                for ixN in range(N):
                    xAdj=np.copy(X)
                    xAdj[ixN]=xAdj[ixN]+h
                    gx2=self.EV_G_ITER(xAdj, k_init, n_agents)
                    A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
            return A

    #======================================================================

    
    
    
    
    
    
    
    
    
            
            
            
    
    
    
    
    
    
