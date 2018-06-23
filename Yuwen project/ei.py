""" 
source: https://github.com/lefnire/tforce_btc_trader/blob/master/gp.py
Bayesian optimisation of loss functions.
"""

import numpy as np
import pickle

from scipy.stats import norm
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
discretized parameter space
'''
#using preprocessed parameter space
pkl_file = open('saffa_sample_j_c.pkl', 'rb')
grid_points = pickle.load(pkl_file)
print(grid_points.shape)
pkl_file.close()
#generate uniform parameter space
'''
num_hyperparam = 3
steps = 40
p_bounds = np.array([[-1, 1], [-1, 1]])
grid_1d = []
for i in range(0, num_hyperparam-1):
    tmp = np.logspace(p_bounds[i,0], p_bounds[i,1], steps)
    grid_1d.append(tmp)
 
grid = np.meshgrid(*grid_1d)
grid_points = np.zeros((steps**(num_hyperparam-1), num_hyperparam-1))
for i in range(0, len(grid[0].ravel())):
    for ith_param in range(0, num_hyperparam-1):
        grid_points[i, ith_param] = grid[ith_param].ravel()[i]
'''        
def expected_improvement(x, x_max, gaussian_process, latent_function, n_params=1):
    """
    Arguments:
    ----------
        x: The point for which the expected improvement needs to be computed.
        x_max: Existing best weights
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        latent_function: Numpy array.
            Numpy array that contains latent function value f(x) of each weight in dataset
        n_params: int.
            Dimension of the weights
    """

    x_to_predict = x.reshape(-1, n_params)
    x_max = x_max.reshape(-1, n_params)
    #approximated predictive gaussian model
    tmp_x = np.concatenate((x_to_predict,x_max), axis = 0)
    K1, K2 = gaussian_process._computeCorrelations(tmp_x)
    mu = np.dot(np.dot(K2.T, np.linalg.pinv(gaussian_process.R)), gaussian_process.Y)
    tt = np.linalg.pinv(np.eye(len(gaussian_process.X))+np.dot(gaussian_process.Lambda, gaussian_process.R))
    cov = K1 - np.dot(np.dot(K2.T,tt),np.dot(gaussian_process.Lambda, K2))
    #bivariate covariance for EI calculation
    sigma = cov[0,0] + cov[1,1] - 2*cov[0,1]
    
    #select existing maximal latent function value
    loss_optimum = np.max(latent_function)
    
    # calculate EI value
    with np.errstate(divide='ignore'):
        if sigma > 0:
            Z = (np.asscalar(mu[0]) - loss_optimum) / sigma
            expected_improvement = (np.asscalar(mu[0]) - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        elif sigma == 0:
            expected_improvement = 0
        else:
            input('sigma <0')
    
    '''    
    print('K1', K1)
    print('K2', K2)
    print('Z', Z)
    print(gaussian_process.Y)
    print('cov', cov)    
    print('sigma', sigma)
    print('mu', mu)
    print('max y', np.max(gaussian_process.Y))
    print('ei ', expected_improvement)
    '''
    return expected_improvement


def sample_next_hyperparameter(acquisition_func, x_max, gaussian_process, latent_function,
                               bounds=np.atleast_2d([0, 100])):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        latent_function: Numpy array.
            Numpy array that contains latent function value of each weight in dataset
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.

    """
    best_x = None
    best_acquisition_value = -np.inf
    n_params = bounds.shape[0]
    
    n_samples = grid_points.shape[0]
    ei_grid = np.zeros(n_samples)
    print(ei_grid.shape)
    for i in range(0, n_samples):
        ei_value = expected_improvement(grid_points[i,:], x_max, gaussian_process, latent_function, n_params)
        ei_grid[i] = ei_value
    #selecting best_x
        '''
        if ei_value > best_acquisition_value:
            best_acquisition_value = ei_value
            best_x = grid_points[i,:]
        '''
    #selecting best_x from distribution based on normalized EI value
    norm_ei_grid = ei_grid/np.sum(ei_grid)
    best_x = grid_points[np.random.choice(len(grid_points), 1, p = norm_ei_grid), :]

    print('best_x: ', best_x)
    avg_ei = np.mean(ei_grid)
    print('Averaged_EI: ', avg_ei)
    #record averaged EI value
    f = open('./EI.txt', 'a')
    f.write(str(avg_ei)+'\n')
    f.close()
    
    #EI plot for 2d
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(grid_points[:,0], grid_points[:,1], ei_grid, marker='o')
    plt.show()
    return best_x
