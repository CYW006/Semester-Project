from time import time
# from copy import copy
import pdb

from numpy import *
import numpy as np
import h5py
import pickle
import scipy.stats as st

from numpy.linalg import LinAlgError
from scipy.optimize import fmin_bfgs, minimize
from kernel import GaussianKernel_ard
from QuadCameraContourTrajectory import QuadCameraContourTrajectory
from generate_kml_file import generate_kml_file
from ei import sample_next_hyperparameter, expected_improvement


#############################################################################
# this implementation of erf, cdf and pdf is substantially faster than
# the scipy implementation (a C implementation would probably be faster yet)
#############################################################################
#
# from: http://www.cs.princeton.edu/introcs/21function/ErrorFunction.java.html
# Implements the Gauss error function.
#   erf(z) = 2 / sqrt(pi) * integral(exp(-t*t), t = 0..z)
#
# fractional error in math formula less than 1.2 * 10 ^ -7.
# although subject to catastrophic cancellation when z in very close to 0
# from Chebyshev fitting formula for erf(z) from Numerical Recipes, 6.2

"""
Initialization
"""
tra_h5 = 'saffa.h5'    #name of trajectory
goal_time = 25                  #toy fixed-time

#grid_points are weights
trajectory_kml = {0: '6_snap_1.h5' , 1: '8_horus_1.h5', 2: 'p3_snap_1.h5'}
noise_low_thd = 0.1     #lower boundary for noise
noise_up_thd = 2        #upper boundary for noise
iso_low_thd = 0.1       #lower boundary for kernel hyperparameters
iso_up_thd = 10         #upper boundary for kernel hyperparameters
init_num = 1            #num of initial preferences
num_hyperparam = 3      #d+1 hyperparameter
num_iter = 50           #num of iteration for optimization
mode = 1                # mode 0: random          mode: 1 GP
"""
grid_points are parameter space
"""
#preprocessed parameter space
pkl_file = open('saffa_sample_j_c.pkl', 'rb')
grid_points = pickle.load(pkl_file)
print(grid_points.shape)
pkl_file.close()

#uniform parameter space
'''
steps = 40              #num of discretized parameter space in one dimension
p_bounds = np.array([[0.1, 50], [0.1, 50]])    #boundary of weights
grid_1d = []
for i in range(0, num_hyperparam - 1):
    tmp = np.linspace(p_bounds[i,0], p_bounds[i,1], steps)
    grid_1d.append(tmp)
    
grid = np.meshgrid(*grid_1d)

grid_points = np.zeros((steps**(num_hyperparam-1), (num_hyperparam-1)))
for i in range(0, len(grid[0].ravel())):
    for ith_param in range(0, (num_hyperparam-1)):
        grid_points[i, ith_param] = grid[ith_param].ravel()[i]
'''
print(grid_points.shape)

# pdf and cdf of gaussian distribution
def erf(z):
    t = 1.0 / (1.0 + 0.5 * abs(z))
    # use Horner's method
    ans = 1 - t * exp( -z*z -  1.26551223 +
                                t * ( 1.00002368 +
                                t * ( 0.37409196 + 
                                t * ( 0.09678418 + 
                                t * (-0.18628806 + 
                                t * ( 0.27886807 + 
                                t * (-1.13520398 + 
                                t * ( 1.48851587 + 
                                t * (-0.82215223 + 
                                t * ( 0.17087277))))))))))
    if z >= 0.0:
        return ans
    else:
        return -ans

def CDF(x):
    return 0.5 * (1 + erf((x) * 0.707106))
    
def PDF(x):
    return  exp(-(x**2/2)) * 0.398942


class GaussianProcess(object):

    def __init__(self, kernel, X=None, Y=None, prior=None, noise=.1, G=None):
        """
        Initialize a Gaussian Process.
        
        kernel:       kernel object to use
        noise:        noise hyperparameter sigma^2_n
        X:            weights
        Y:            laten function value for corresponding weights
        """
        self.kernel = kernel
        self.prior = prior
        self.noise = noise
        
        self.R = None
        
        if (X is None and Y is not None) or (X is not None and Y is None):
            raise ValueError
            
        self.X = zeros((0,0))
        self.Y = zeros((0))
        
        self.G = None
        
    def _computeCorrelations(self, X):
        """ compute correlations between data """
        M, (N, D) = len(self.X), X.shape
        r = eye(N, dtype=float)*(1 + self.noise)
        m = empty((M,N))
        
        for i in range(N):
            for j in range(i): 
                r[i,j] = r[j,i] = self.kernel.cov(X[i], X[j])
                
        for i in range(M):
            for j in range(N): 
                m[i,j] = self.kernel.cov(self.X[i], X[j])
                
        return r, m        

    def posterior(self, X, getvar=True):
        """ Get posterior mean and variance for a point X. """
        if len(self.X)==0:
            if self.prior is None:
                if getvar:
                    return 0.0, 1.0
                else:
                    return 0.0
            else:
                if getvar:
                    return self.prior.mu(X), 1.0
                else:
                    return self.prior.mu(X)
        
        X = array(X, copy=False, dtype=float, ndmin=2)
        M, (N,D) = len(self.X), X.shape
        
        m = 0.0
        if self.prior is not None:
            m = self.prior.mu(X)
            assert isscalar(m)
        
        d = self.Y-m
        r = empty((M, N))
        for i in range(M):
            for j in range(N): 
                r[i,j] = self.kernel.cov(self.X[i], X[j])
        
        # calculate the mean.
        Lr = linalg.solve(self.L, r)
        mu = m + dot(Lr.T, linalg.solve(self.L,d))
        
        if getvar:
            # calculate the variance.
            if self.L is None:
                sigma2 = (1 + self.noise) - sum(Lr**2, axis=0)
            else:
                M, (N,D) = len(self.X), X.shape
                r = empty((M, N))
                for i in range(M):
                    for j in range(N): 
                        r[i,j] = self.kernel.cov(self.X[i], X[j])
                Lr = linalg.solve(self.L, r)
                sigma2 = (1 + self.noise) - sum(Lr**2, axis=0)
            sigma2 = clip(sigma2, 10e-8, 100)
        
            return mu[0], sigma2[0]
        else:
            return mu[0]
        
    
    def posteriors(self, X):
        """
        get arrays of posterior values for the array in X
        """
        M = []
        V = []
        for x in X:
            if isscalar(x):
                m, v = self.posterior(array([x]))
            else:
                m, v = self.posterior(x)
            M.append(m)
            V.append(v)
        return array(M), array(V)     
        
    def mu(self, x):
        """
        get posterior mean for a point x
        
        NOTE: if you are getting the variance as well, this is less efficient
        than using self.posterior()
        """
        return self.posterior(x, getvar=False)
        
    
    
class PrefGaussianProcess(GaussianProcess):
    """
    Like a regular Gaussian Process, but trained on preference data.  Note
    that you cannot (currently) add non-preference data.  This is because I
    haven't gotten around to it, not because it's impossible.
    """
    def __init__(self, kernel, prefs=None, bounds = None, **kwargs):
        super(PrefGaussianProcess, self).__init__(kernel, **kwargs)
        
        self.preferences = np.empty((0,0,0))
        self.Lambda = None
        self.loss = []
        self.bounds = bounds
        
        if prefs is not None:
            self.addPreferences(prefs)
        else:
            self.build_prior()

    #based on EI function to find the next parameter
    def find_newpoint(self): 
             
        if (not mode):
            repeat = 1
            while (repeat):
                repeat = 0
                next_point = grid_points[np.random.randint(grid_points.shape[0]),:]
                for tmp in self.X:
                    if (next_point==tmp).all():
                        repeat = 1
                        input('repeat')
        else:
            next_point =  sample_next_hyperparameter(expected_improvement, self.X[np.argmax(self.Y)], self, self.Y, bounds= self.bounds)
       
        return next_point
    
    #now just want to approximate -(x-2)^2 x=2; further design an user-input
    def build_preference(self, x_new):
        #input: x_new: new parameter
        #output: preference (data type: np_array2d)
        def f(x):
            return (x[0] - 2)**2 + (x[1] - 1)**2
        x_new = np.ravel(x_new)
        x_min = self.X[np.argmin(f(self.X))]

        pref = np.zeros((2,1,len(x_new)))
        if (f(x_new) < f(x_min)):
            pref[0,0,:] = x_new
            pref[1,0,:] = x_min
        elif (f(x_new) > f(x_min)):
            pref[0,0,:] = x_min
            pref[1,0,:] = x_new
        else:       #no preference
            print('finishes')
            return np.array([])
        return pref
  
    def build_prior(self):
        '''
        initialize two groups of randomly selected weights
        '''
        for i in range(0, init_num):
            min_sample = grid_points[np.random.randint(grid_points.shape[0]),:]
            new_sample = grid_points[np.random.randint(grid_points.shape[0]),:]
            #new_sample = init_sample[i,:]
            #select trajectory
            trajectory_name = tra_h5
            #generate new videos
            err_new, err_min = self.generate_compared_videos(new_sample, min_sample, trajectory_name)
            
            print(new_sample, '1 time is: '+str(err_new))
            print(min_sample, '2 time is: '+str(err_min))
            print(trajectory_name)            
            pref = np.zeros((2,1,grid_points.shape[1]))
    
            pref_decision = input("choose preference: (1 for trajectory_1 and 2 for trajectory_2) ")
            if (pref_decision == '1'):
                pref[0,0,:] = np.array(new_sample).reshape((1,-1))
                pref[1,0,:] = np.array(min_sample).reshape((1,-1))
                min_sample = new_sample
            else:
                pref[0,0,:] = np.array(min_sample).reshape((1,-1))
                pref[1,0,:] = np.array(new_sample).reshape((1,-1))
            self.addPreferences(pref)
            print(self.preferences.shape)
        return
    
    def addPreferences(self, prefs, showPrefLikelihood=False):
        """
        Add a set of preferences to the GP and optimize latent function and hyperparameters.

        prefs:  sequence of preference triples (xv, xu) where xv
                        is a datum preferred to xu
                        
                        first dimension is 2 (comparison), second is num of preferences, third is num of parameters per sample
        """

        # add new preferences
        if len(self.preferences) == 0:
            self.preferences = prefs
        else:
            self.preferences = np.concatenate((self.preferences,prefs),axis=1)

        x2ind = {}#dictionary
        ind = 0
        #find all X
        for i in range(0,self.preferences.shape[1]):
            v = tuple(self.preferences[0,i,:])
            u = tuple(self.preferences[1,i,:])
            if v not in x2ind:
                x2ind[v] = ind
                ind += 1
            if u not in x2ind:
                x2ind[u] = ind
                ind += 1
        
        #generate new X
        newX = np.zeros((0,prefs.shape[2]))
        for x in x2ind:
            newX = np.concatenate((newX, np.asarray(x).reshape(1,-1)), axis = 0)
        
        # update X, R, L
        self.X = np.zeros((0,0))
        r, m = self._computeCorrelations(newX)
        self.X = newX
        self.R = r          #covariance matrix of prior
        self.L = np.linalg.cholesky(self.R)
        
        def S(x, useC=False):
            """
            the MAP latent function f(x) to be optimized
            """
            logCDFs = 0.
            sigma = sqrt(self.noise)
            epsilon = 1e-10
            Z = sqrt(2) * sigma
            for i in range(0,self.preferences.shape[1]):
                s1 = x2ind[tuple(self.preferences[0,i,:])]
                s2 = x2ind[tuple(self.preferences[1,i,:])]
                logCDFs += log(CDF((x[s1]-x[s2])/Z)+epsilon)
            
            Lx = linalg.solve(self.L, x)
            #print(Lx.shape)
            val = -logCDFs + dot(Lx, Lx)/2.0
            if not isfinite(val):
                print ('non-finite val!')
                pdb.set_trace()
            # print '\n***** val =', val
            return val

        def Q(var):
            """
            optimize d+1 hyperparameters
            """  
            logCDFs = 0.
            sigma = var[0]
            epsilon = 1e-10
            Z = sqrt(2) * sigma
            for i in range(0,self.preferences.shape[1]):
                s1 = x2ind[tuple(self.preferences[0,i,:])]
                s2 = x2ind[tuple(self.preferences[1,i,:])]
                logCDFs += log(CDF((self.Y[s1]-self.Y[s2])/Z)+epsilon)
            
            #prior information
            locs = [0.3, 5, 5]
            sigmas = [2, 2, 2]
            nus = [2, 10, 10.]
            logPriors = log(st.lognorm.pdf(var[0], 1.5, scale=np.exp(0.5)))
            #logPriors = 0
            for j in range(0, len(var)):
                #logPriors += log(st.t.pdf(var[j], df=nus[j], loc=locs[j], scale=sigmas[j]))
                logPriors += log(st.t.pdf(var[j], df=nus[j], loc=locs[j], scale=sigmas[j]))
            
            self.kernel = GaussianKernel_ard(var[1:])                
            newX = self.X
            r, m = self._computeCorrelations(newX)
            self.R = r
            self.L = np.linalg.cholesky(self.R)
            Lx = linalg.solve(self.L, self.Y.ravel())

            #C = eye(len(self.X), dtype=float) * 5
            self.Lambda = zeros((len(self.X), len(self.X)))
            for i in range(len(self.X)):
                for j in range(len(self.X)):
                    for k in range(self.preferences.shape[1]):
                        r = self.preferences[0,k,:]
                        c = self.preferences[1,k,:]
                        alpha = 0
                        if all(r==self.X[i]) and all(c==self.X[j]):
                            alpha = -1
                        elif all(r==self.X[j]) and all(c==self.X[i]):
                            alpha = -1
                        elif all(r==self.X[i]) and i==j:
                            alpha = 1
                        elif all(c==self.X[i]) and i==j:
                            alpha = 1
                        s1 = x2ind[tuple(self.preferences[0,k,:])]
                        s2 = x2ind[tuple(self.preferences[1,k,:])]
                        if alpha != 0:
                            # print 'have an entry for %d, %d!' % (i,j)
                            Z_ = (self.Y[s1]-self.Y[s2]) / (sqrt(2)*sqrt(sigma))
                            # print '\td=',d
                            cdf = CDF(Z_)
                            pdf = PDF(Z_)
                            if cdf < 1e-10:
                                cdf = 1e-10
                            if pdf < 1e-10:
                                pdf = 1e-10
                            self.Lambda[i,j] += alpha / (2*sigma) * (pdf**2/cdf**2 + Z_ * pdf/cdf)
           
            val = -logCDFs + dot(Lx, Lx)/2.0 + 0.5*log(np.linalg.det(np.eye(len(self.X)) + dot(self.R, self.Lambda))) - logPriors #maximize P(D|theta)
            if not isfinite(val):
                print ('non-finite val!')
                pdb.set_trace()
            # print '\n***** val =', val
            return val

        # use existing Ys as starting point for optimizer
        if len(self.Y) > 0:
            #x_new in new preference has been used before, there is no new sample be added
            start = np.concatenate((self.Y, np.atleast_2d(np.random.uniform(np.min(self.Y), np.max(self.Y), size = (len(x2ind)-len(self.Y),1)))),axis=0)
        else:   #initialization for the first iteration
            start = np.empty((len(x2ind),1))
            for i in range(0,len(x2ind)):
                start[i] = np.random.uniform(-10,10)

        '''
        optimize latent function f(x), randomdize 3 times
        '''                
        res = minimize(S, start, method='BFGS', options={'maxiter': num_iter})  
        self.Y = res['x'].reshape((-1,1))      
        best_noise_obj_value = res['fun']
        
        for i in range(0,3):
            if len(self.Y) > 0:
                start = np.atleast_2d(np.random.uniform(np.min(self.Y), np.max(self.Y), size = (len(x2ind),1)))
            else:   #initialization for the first iteration
                for i in range(0,len(x2ind)):
                    start[i] = np.random.uniform(-50,50)
            res = minimize(S, start, method='BFGS', options={'maxiter': num_iter})  
            if (best_noise_obj_value > res['fun']):
                best_noise_obj_value = res['fun']
                self.Y = res['x'].reshape((-1,1))
        
        '''
        optimize hyperparameters, randomdize 3 times
        '''
        #self.noise = fmin_tnc(Q, self.noise, bounds=((0.1, 5),), approx_grad=True)[0]  #optimize hyperparameter
        hyperparam = np.empty((num_hyperparam,1))
        hyperparam[0] = self.noise
        hyperparam[1:] = self.kernel._theta.reshape((-1,1))
        
        res = minimize(Q, hyperparam, bounds=((noise_low_thd, noise_up_thd), (iso_low_thd, iso_up_thd), (iso_low_thd, iso_up_thd)), options={'maxiter': num_iter})
        hyperparam = res['x']
        self.noise = hyperparam[0]
        self.kernel = GaussianKernel_ard(hyperparam[1:].ravel())
        best_noise_obj_value = res['fun']
        
        for i in range(3):
            hyperparam = np.empty((num_hyperparam,1))
            hyperparam[0] = np.random.uniform(low = noise_low_thd, high = noise_up_thd)
            for j in range(1, len(hyperparam.ravel())):
                hyperparam[j] = np.random.uniform(low = iso_low_thd, high = iso_up_thd)
            res = minimize(Q, hyperparam, bounds=((noise_low_thd, noise_up_thd), (iso_low_thd, iso_up_thd), (iso_low_thd, iso_up_thd)), options={'maxiter': num_iter})
            if (best_noise_obj_value > res['fun']):
                best_noise_obj_value = res['fun']
                hyperparam = res['x']
                self.noise = hyperparam[0]
                self.kernel = GaussianKernel_ard(hyperparam[1:].ravel() )       
         
        print('noise:', self.noise)
        print('kernel param', self.kernel._theta)
        print('obj_value: ', best_noise_obj_value)
                
        # now we can learn the Lambda matrix
        self.Lambda = zeros((len(self.X), len(self.X)))

        for i in range(len(self.X)):
            for j in range(len(self.X)):
                for k in range(self.preferences.shape[1]):
                    r = self.preferences[0,k,:]
                    c = self.preferences[1,k,:]
                    alpha = 0
                    if all(r==self.X[i]) and all(c==self.X[j]):
                        alpha = -1
                    elif all(r==self.X[j]) and all(c==self.X[i]):
                        alpha = -1
                    elif all(r==self.X[i]) and i==j:
                        alpha = 1
                    elif all(c==self.X[i]) and i==j:
                        alpha = 1
                    if alpha != 0:
                        # print 'have an entry for %d, %d!' % (i,j)
                        Z = (self.mu(r)-self.mu(c)) / (sqrt(2)*sqrt(self.noise))
                        # print '\td=',d
                        cdf = CDF(Z)
                        pdf = PDF(Z)
                        if cdf < 1e-10:
                            cdf = 1e-10
                        if pdf < 1e-10:
                            pdf = 1e-10
                        self.Lambda[i,j] += alpha / (2*self.noise) * (pdf**2/cdf**2 + Z * pdf/cdf)
        try:
            self.L = linalg.cholesky(self.R+linalg.inv(self.Lambda))
        except LinAlgError:
            print ('[addPreferences] GP.C matrix is ill-conditioned, adding regularizer delta = 1')
            for i in range(10):
                self.Lambda += eye(len(self.X))
                try:
                    self.L = linalg.cholesky(self.R+linalg.inv(self.Lambda))
                except LinAlgError:
                    print ('[addPreferences] GP.C matrix is ill-conditioned, adding regularizer delta = %d' % (i+2))
                else:
                    break
                
        print ('[addPreferences] checking pref pairs')
        for i in range(0,self.preferences.shape[1]):
            r = tuple(self.preferences[0,i,:])
            c = tuple(self.preferences[1,i,:])
            
            #if latent function value not consistent with preferences
            if self.Y[x2ind[r]] <= self.Y[x2ind[c]]:
                print ('   FAILED!  %.2f ! > %.2f'  % (self.Y[x2ind[r]], self.Y[x2ind[c]]))
                print ('which preference:', r, c) 
                
                switch = self.retest(r, c)
                if (switch):
                    self.preferences[0,i,:] = c
                    self.preferences[1,i,:] = r
                    #print(self.Y[x2ind[r]],self.Y[x2ind[c]])
                    #reoptimize self.Y
                    self.Y[x2ind[r]],self.Y[x2ind[c]] = self.Y[x2ind[c]], self.Y[x2ind[r]]
                    self.Y = fmin_bfgs(S, self.Y, disp=0).reshape(-1,1) 
                    #print(self.Y[x2ind[r]],self.Y[x2ind[c]]) 
                else:
                    #reinforce this preference
                    #input("reinforce preference: ")
                    reinforcement = np.zeros((2,1,grid_points.shape[1]))
                    reinforcement[0,0,:] = self.preferences[0,i,:]
                    reinforcement[1,0,:] = self.preferences[1,i,:]           
                    self.preferences = np.concatenate((self.preferences, reinforcement),axis=1)
         
    def build_preference_ge(self):
        '''
        build new preference for each iteration
        '''
        #select trajectory
        trajectory_name = tra_h5
        print(trajectory_name)
        # finding new weights
        new_x = self.find_newpoint()
        
        weights_new = new_x.ravel()
        weights_min = self.X[np.argmax(self.Y)]
        
        err_new, err_min = self.generate_compared_videos(weights_new, weights_min, trajectory_name)
        
        print(weights_new, 'new_sample time is: '+str(err_new))
        print(weights_min, 'premax_sample time is: '+str(err_min))
        
        pref = np.zeros((2,1,grid_points.shape[1]))
        
        if (err_new < err_min):
            pref[0,0,:] = np.array(weights_new).reshape((1,-1))
            pref[1,0,:] = np.array(weights_min).reshape((1,-1))
        elif (err_new > err_min):
             pref[0,0,:] = np.array(weights_min).reshape((1,-1))
             pref[1,0,:] = np.array(weights_new).reshape((1,-1))
        else:       #no preference
            print('finishes')
            return np.array([])
        '''
        pref_decision = input("choose preference: (1 for new setting parameters and 2 for pre_max) ")
        if (pref_decision == '1'):
            pref[0,0,:] = np.array([j_w_new, c_w_new]).reshape((1,-1))
            pref[1,0,:] = np.array([j_w_min, c_w_min]).reshape((1,-1))
        elif (pref_decision == '2'):
             pref[0,0,:] = np.array([j_w_min, c_w_min]).reshape((1,-1))
             pref[1,0,:] = np.array([j_w_new, c_w_new]).reshape((1,-1))
        else:       #no preference
            print('finishes')
            return np.array([])
        '''
        self.loss.append(err_new)
        
        return pref
    
    def generate_compared_videos(self, weights_1, weights_2, trajectory_name):
        '''
        generate two videos for comparison
        '''
        trajectory_name = trajectory_name
        print(trajectory_name)

        j_w_1, c_w_1 = weights_1
        lag_weight = 2                                  # weight on e^l
        contour_weight = c_w_1                              # weight on e^c
        angular_weight = 1                              # w_phi and w_psi  
        # learn jerk	
        jerk_weight = j_w_1                                # w_j on position
        angular_jerk_weight = 1                         # w_j on angles
        theta_input_weight = 1                          # weight on v_i
        # not necessary
        min_time_weight = 1                             # w_end
        end_time_weight = 0                            # w_len
        timing_weight = 0                               # w_t
    
        # load trajectory
        keyframes, keyorientations, keytimes, gps_trajectory = load_trajectory(trajectory_name)
        if keyframes.shape[1] > 3:
            keyframes = keyframes[:,0:3]
    
        # specifying weights for optimizer
        options = dict()
        options["keytimes"] = keytimes
        options["lag_weight"] = lag_weight
        options["contour_weight"] = contour_weight
        options["gimbal_weight"] = angular_weight
        options["smoothness_weight"] = jerk_weight
        options["angular_smoothness_weight"] = angular_jerk_weight
        options["theta_input_weight"] = theta_input_weight
        options["min_time_weight"] = min_time_weight
        options["end_time_weight"] = end_time_weight
        options["timing_weight"] = timing_weight
    
        trajGen = QuadCameraContourTrajectory(keyframes, keyorientations, options)
        trajectory, keytimes = trajGen.generate_trajectory()
        end_time = trajectory[trajGen.endTimeStateIndex, -1];
        err_1 = np.absolute(end_time -goal_time) 
        
        # create kml-file and save to directory "kmls"
        generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name, '_new')
        
        j_w_2, c_w_2 =  weights_2
        
        contour_weight = c_w_2                             # w_len
        jerk_weight = j_w_2
        options["contour_weight"] = contour_weight
        options["min_time_weight"] = min_time_weight
        options["smoothness_weight"] = jerk_weight
        trajGen = QuadCameraContourTrajectory(keyframes, keyorientations, options)
        trajectory, keytimes = trajGen.generate_trajectory()
        end_time = trajectory[trajGen.endTimeStateIndex, -1];
        err_2 = np.absolute(end_time - goal_time)
        # create kml-file and save to directory "kmls"
        generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name, '_premax')
        
        return err_1, err_2
        
    def retest(self, weights_1, weights_2):
        '''
        retest the preference that is not consistent with latent function value at this stage
        weights_1 and weights_2 in this preference
        '''
        trajectory_name = tra_h5
        #trajectory_name = trajectory_kml[np.random.randint(len(trajectory_kml))]
        
        err_1, err_2 = self.generate_compared_videos(weights_1, weights_2, trajectory_name)
        
        print(weights_1, 'weights_1 time is: '+str(err_1))
        print(weights_2, 'weights_2 time is: '+str(err_2))
        
        if (err_1 < err_2):
            switch = False
        elif (err_1 > err_2):
            switch = True
        else:       #no preference
            print('no preference')
            switch = False 
        '''
        pref_decision = input("choose preference: (1 for weights_1 and 2 for weights_2) ")
        if (pref_decision == '1'):
            switch = False
        elif (pref_decision == '2'):
            switch = True
        else:       #no preference
            print('no preference')
            switch = False        
        '''
        return switch
        
def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory        
