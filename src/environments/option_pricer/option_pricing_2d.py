import numpy as np
from src.environments.option_pricer import binomial
from .MonteCarlo import MonteCarlo
from matplotlib import pyplot as plt
from src.utils import random_hypercube_samples

# Gaussian Process Regression in 2D
# time_to_maturity for the option
# strike price
# n_trials and n_steps for the Monte Carlo simulation 
# vol -->  volatility values
# min_vol --> minimum value of the volatility
# max_vol --> maximum value of the volatility
# Option type --> can be 'c' for CALL and 'p' for PUT options
# is_american --> True if it is an American option, false otherwise
def heston_option_pricing_2d(time_to_maturity, strike, n_trials, n_steps, vol, min_vol, max_vol,o_type,is_american,kappa, bounds, grid_test=False):
    risk_free_rate = 0.1
    dividend = 0.3

    # Only for BlackScholes model
    volatility = 0

    # stock price
    stock_price = 1


    # Mean reversion speed
    if(not kappa):
        kappa = 3.00 - 0.3*4

    # Theta =  Gamma(in the paper)
    theta = 3/kappa*0.03

    rho = -0.6

    # volatility of volatility
    xi = 0.6

    X = []
    y = []

    # Generating training data
    for t in time_to_maturity:
        for v in vol:
            # Initial Variance (hidden state) 
            # Volatility squared
            V0 = v**2
            mc = MonteCarlo(S0=stock_price,K=strike,T=t,r=risk_free_rate,q=dividend,sigma=volatility,
                        kappa=kappa,theta=theta,xi=xi,rho=rho,V0=V0,underlying_process="Heston model")
            price_matrix = mc.simulate(n_trials=n_trials,n_steps=n_steps,boundaryScheme="Higham and Mao")
            if (is_american):
                mc.LSM(option_type=o_type,func_list=[lambda x: x**0, lambda x: x],onlyITM=False,buy_cost=0.0,sell_cost=0.0)
                # print(mc.american_values_matrix)
            price = mc.MCPricer(option_type=o_type, isAmerican=is_american) 
            # print(price)
            X.append([v,t])
            y.append(price)

    y = np.array(y)[:, np.newaxis]

    #  Number of test points
    n = 50

    X = np.array(X)
    # print(X)
    # X = X.transpose()
    # print(X)
    # X = np.array(X).reshape(-1,1)
    # print(X)

    if grid_test:
        # Points we are going to make prediction at
        x1 = np.linspace(min_vol,max_vol,n)
        x2 = np.linspace(time_to_maturity[0],time_to_maturity[-1],n)
        # lin = np.linspace(-lim, lim, res)

        # x1.shape = (50, 50)
        xx, xt = np.meshgrid(x1, x2)
        # xx.shape = (2500, 2)
        X_test = np.vstack((xx.flatten(), xt.flatten())).T
    else:
        x1 = None
        x2 = None
        X_test = random_hypercube_samples(bounds, 2500, rng=np.random.RandomState(42))

    YT=[]
    # Compute real price for testPoints
    for x in X_test:
        [t,v] = x
        # np.random.seed(1)
        V0 = v**2
        mc = MonteCarlo(S0=stock_price,K=strike,T=t,r=risk_free_rate,q=dividend,sigma=volatility,
                    kappa=kappa,theta=theta,xi=xi,rho=rho,V0=V0,underlying_process="Heston model")
        price_matrix = mc.simulate(n_trials=n_trials,n_steps=n_steps,boundaryScheme="Higham and Mao")
        if (is_american):
            mc.LSM(option_type=o_type,func_list=[lambda x: x**0, lambda x: x],onlyITM=False,buy_cost=0.0,sell_cost=0.0)
        price = mc.MCPricer(option_type=o_type, isAmerican=is_american) 
        YT.append(price)

    YT = np.array(YT)[:,np.newaxis]
    return [X,y,x1,x2,X_test,YT]
