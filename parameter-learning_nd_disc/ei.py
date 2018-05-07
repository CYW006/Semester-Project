""" 
source: https://github.com/lefnire/tforce_btc_trader/blob/master/gp.py
Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def expected_improvement(x, x_max, kernel, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.posteriors(x_to_predict)
    '''
    add covariance between predicted_new_point and existing max point
    '''
    sigma += kernel.cov(x_max,x_max) -2*kernel.cov(x_to_predict,x_max)
    
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma.reshape((-1, 1))
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma.reshape((-1, 1)) * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -expected_improvement


def sample_next_hyperparameter(acquisition_func, x_max, kernel, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=np.atleast_2d([0, 10]), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    
    grid_1d = []
    for i in range(0, n_params):
        tmp = np.linspace(bounds[i,0], bounds[i,1],50)
        grid_1d.append(tmp)
        
    grid = np.meshgrid(*grid_1d)

    grid_points = np.zeros((50**n_params, n_params))
    for i in range(0, len(grid[0].ravel())):
        for ith_param in range(0, n_params):
            grid_points[i, ith_param] = grid[ith_param].ravel()[i]
    '''
    ei_grid = expected_improvement(grid_points, x_max, kernel, gaussian_process, evaluated_loss, greater_is_better, n_params)
    ei_grid = np.atleast_2d(ei_grid.reshape((100,100)))
    print(ei_grid.shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot_surface(grid[0], grid[1], -ei_grid, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.scatter(grid[0], grid[1], -ei_grid, marker='o')
#    ax.set_zlim3d(-1, 1)
#    plt.imshow(ei_grid)
    plt.show()
    print('max', -np.min(ei_grid))
    '''
    
    
    ei_grid = np.zeros(50**n_params)
    for i in range(0, len(grid[0].ravel())):
        '''
        x = np.empty_like(x_max)
        for ith_param in range(0, n_params):
            x[ith_param] = grid[ith_param].ravel()[i]
        
        x = x.reshape(1, n_params)
        '''
        ei_value = expected_improvement(grid_points[i,:], x_max, kernel, gaussian_process, evaluated_loss, greater_is_better, n_params)
        ei_grid[i] = ei_value
        if ei_value < best_acquisition_value:
            best_acquisition_value = ei_value
            best_x = grid_points[i,:]
        
    print('best_value: ', -best_acquisition_value)
    print('best_x: ', best_x)
    ei_grid = np.atleast_2d(ei_grid.reshape((50,50)))
    print(ei_grid.shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot_surface(grid[0], grid[1], -ei_grid, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.scatter(grid[0], grid[1], -ei_grid, marker='o')
#    ax.set_zlim3d(-1, 1)
#    plt.imshow(ei_grid)
    plt.show()
    '''
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(x_max, kernel, gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = np.ndarray.flatten(res.x)
    '''
    return best_x