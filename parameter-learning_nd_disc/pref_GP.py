from time import time
# from copy import copy
import pdb

from numpy import *
import numpy as np
import scipy as sci
import h5py
import sklearn.gaussian_process as gp

from scipy.stats import norm
from numpy.linalg import inv, LinAlgError
from scipy.optimize import fmin_bfgs, fmin_tnc, minimize
from kernel import GaussianKernel_ard, GaussianKernel_iso, MaternKernel3, MaternKernel5
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

    def __init__(self, kernel, X=None, Y=None, prior=None, noise=.1, gnoise=1e-4, G=None):
        """
        Initialize a Gaussian Process.
        
        @param kernel:       kernel object to use
        @param prior:        object defining the GP prior on the mean.  must 
                             be a descendant of GPMeanPrior
        @param noise:        noise hyperparameter sigma^2_n
        @param X:            initial training data
        @param Y:            initial observations
        """
        self.kernel = kernel
        self.prior = prior
        self.noise = noise
        self.gnoise = array(gnoise, ndmin=1)
        
        self.R = None
        
        if (X is None and Y is not None) or (X is not None and Y is None):
            raise ValueError
            
        self.X = zeros((0,0))
        self.Y = zeros((0))
        
        self.G = None
        
        self.name = 'GP'            # for legend
        self.starttime = time()     # for later analysis

        if X is not None:
            self.addData(X, Y)
            
        self.augR = None
        self.augL = None
        self.augX = None
        
        # mostly for testing/logging
        self.selected = None
        self.endtime = None
        
        # if prior is None:
        #     print 'prior is None'
        # else:
        #     print 'prior is NOT None'
        #     
        # if self.prior is None:
        #     print 'self.prior is None'
        # else:
        #     print 'self.prior is NOT None'
        
        
    def _computeCorrelations(self, X):
        """ compute correlations between data """
        M, (N, D) = len(self.X), X.shape
        r = eye(N, dtype=float) + self.noise
        m = empty((M,N))
        
        for i in range(N):
            for j in range(i): 
                r[i,j] = r[j,i] = self.kernel.cov(X[i], X[j])
                
        for i in range(M):
            for j in range(N): 
                m[i,j] = self.kernel.cov(self.X[i], X[j])
                
        return r, m
        
    def _computeAugCorrelations(self, X):
        """ compute correlations between data """

        M, (N,D) = len(self.augX), X.shape
        r = eye(N, dtype=float) + self.noise
        m = empty((M,N))
        
        for i in range(N):
            for j in range(i): 
                r[i,j] = r[j,i] = self.kernel.cov(X[i], X[j])
                
        for i in range(M):
            for j in range(N): 
                m[i,j] = self.kernel.cov(self.augX[i], X[j])
                
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
            if self.augL is None:
                sigma2 = (1 + self.noise) - sum(Lr**2, axis=0)
            else:
                M, (N,D) = len(self.augX), X.shape
                r = empty((M, N))
                for i in range(M):
                    for j in range(N): 
                        r[i,j] = self.kernel.cov(self.augX[i], X[j])
                Lr = linalg.solve(self.augL, r)
                sigma2 = (1 + self.noise) - sum(Lr**2, axis=0)
            sigma2 = clip(sigma2, 10e-8, 10)
        
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
            
            
    def negmu(self, x):
        """
        needed occasionally for optimization
        """
        nm = -self.mu(x)
        # if self.prior is not None and len(self.X)==0:
        #     print 'no data, using prior = %.4f'%nm
        return nm
        
        
    def addData(self, X, Y, G=None):
        """
        Add new data to model and update. 

        We assume that X is an (N,D)-array, Y is an N-vector, and G is either
        an (N,D)-array or None. Further, if X or G are a single D-dimensional
        vector these will be interpreted as (1,D)-arrays, i.e. one observation.
        """
        X = array(X, copy=False, dtype=float, ndmin=2)
        Y = array(Y, copy=False, dtype=float, ndmin=1).flatten()
        G = array(G, copy=False, dtype=float, ndmin=2) if (G is not None) else None

        assert len(Y) == len(X), 'wrong number of Y-observations given'
        assert G is None or G.shape == X.shape, 'wrong number (or dimensionality) of gradient-observations given'
        # print '(', len(self.X), self.G, G, ')'
        # assert not (len(self.X) > 0 and self.G is not None and G is None), 'gradients must either be always or never given'

        # this just makes sure that if we used the default gradient noise for
        # each dimension it gets lengthened to the proper size.
        if len(self.X) == 0 and len(self.gnoise) == 1: 
            self.gnoise = tile(self.gnoise, X.shape[1])

        # compute the correlations between our data points.
        r, m = \
            self._computeCorrelations(X) if (G is None) else \
            self._computeCorrelationsWithGradients(X)

        if len(self.X) == 0:
            self.X = copy(X)
            self.Y = copy(Y)
            self.G = copy(G) if (G is not None) else None
            self.R = r
            self.L = linalg.cholesky(self.R)
        else:
            self.X = r_[self.X, X]
            self.Y = r_[self.Y, Y]
            self.G = r_[self.G, G] if (G is not None) else None
            self.R = r_[c_[self.R, m], c_[m.T, r]]

            z = linalg.solve(self.L, m)
            d = linalg.cholesky(r - dot(z.T, z))
            self.L = r_[c_[self.L, zeros(z.shape)], c_[z.T, d]]
        # print '\nself.G =', G, ', for which selfG is None is', (self.G is None)
            
            
    def getYfromX(self, qx):
        """
        get the (first) Y value for a given training datum X.  return None if x 
        is not found.
        """
        for x, y in zip(self.X, self.Y):
            if all(qx==x):
                return y
        return None
        
    def done(self, x):
        """
        indication that the GP has been terminated and that a final point has
        been selected (mostly relevant for logging)
        """
        self.selected = x
        self.endtime = time()

        
class PrefGaussianProcess(GaussianProcess):
    """
    Like a regular Gaussian Process, but trained on preference data.  Note
    that you cannot (currently) add non-preference data.  This is because I
    haven't gotten around to it, not because it's impossible.
    """
    def __init__(self, kernel, prefs=None, **kwargs):
        super(PrefGaussianProcess, self).__init__(kernel, **kwargs)
        
        self.preferences = np.empty((0,0,0))
        self.C = None
        self.loss = []
        
        if prefs is not None:
            self.addPreferences(prefs)
    def predict(self):
        return mu, sigma

    #based on EI function to find the next parameter
    def find_newpoint(self, p_bounds): 
        return sample_next_hyperparameter(expected_improvement, self.X[np.argmax(self.Y)], self.kernel, self, self.Y, greater_is_better=True, bounds= p_bounds, n_restarts=50)
     
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
    
    def addPreferences(self, prefs, showPrefLikelihood=False):
        """
        Add a set of preferences to the GP and update.

        @param  prefs:  sequence of preference triples (xv, xu) where xv
                        is a datum preferred to xu
                        
                        first dimension is comparison, second is num of preferences, third is num of parameters per sample
        """

        # add new preferences
        if len(self.preferences) == 0:
            self.preferences = prefs
        else:
            self.preferences = np.concatenate((self.preferences,prefs),axis=1)

        x2ind = {}#dictionary
        ind = 0

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
        
        # update X, R
        self.X = np.zeros((0,0))
        r, m = self._computeCorrelations(newX)
        self.X = newX
        self.R = r
        self.L = np.linalg.cholesky(self.R)

        # use existing Ys as starting point for optimizer
        start = np.empty((len(x2ind),1))
        if len(self.Y) > 0:
            #x_new in new preference has been used before, there is no new sample be added
            if(len(self.Y) == len(x2ind)):
                start = self.Y
            #x_new has never been used
            else:
                start = np.concatenate((self.Y, np.atleast_2d(np.random.uniform(np.min(self.Y), np.max(self.Y)))),axis=0)
        else:   #initialization for the first iteration
            for i in range(0,len(x2ind)):
                start[i] = np.random.uniform(0,5)

        # optimize S to find a good Y
        # self.Y = fmin_bfgs(S, start, args=(prefinds, self.L), epsilon=0.1, maxiter=30, disp=0)
        
        def S(x, useC=False):
            """
            the MAP functional to be minimized
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
        
        self.Y = fmin_bfgs(S, start, disp=0).reshape(-1,1)  #record MAP     
       
        
        def Q(var):
            """
            optimize hyperparameter(eg: noise sigma)
            """
            logCDFs = 0.
            sigma = var
            epsilon = 1e-10
            Z = sqrt(2) * sigma
            for i in range(0,self.preferences.shape[1]):
                s1 = x2ind[tuple(self.preferences[0,i,:])]
                s2 = x2ind[tuple(self.preferences[1,i,:])]
                logCDFs += log(CDF((self.Y[s1]-self.Y[s2])/Z)+epsilon)
            
            Lx = linalg.solve(self.L, self.Y.ravel())

            #C = eye(len(self.X), dtype=float) * 5
            C = zeros((len(self.X), len(self.X)))
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
                            Z_ = (self.mu(r)-self.mu(c)) / (sqrt(2)*sqrt(sigma))
                            # print '\td=',d
                            cdf = CDF(Z_)
                            pdf = PDF(Z_)
                            if cdf < 1e-10:
                                cdf = 1e-10
                            if pdf < 1e-10:
                                pdf = 1e-10
                            C[i,j] += alpha / (2*sigma) * (pdf**2/cdf**2 + Z_ * pdf/cdf)
            
            val = -logCDFs + dot(Lx, Lx)/2.0 + 0.5*log(np.linalg.det(np.eye(len(self.X)) + dot(self.R, C))) #add noise prior
            if not isfinite(val):
                print ('non-finite val!')
                pdb.set_trace()
            # print '\n***** val =', val
            return val
        
        '''
        optimize noise, randomdize 4 times
        '''
        #self.noise = fmin_tnc(Q, self.noise, bounds=((0.1, 5),), approx_grad=True)[0]  #optimize hyperparameter
        res = minimize(Q, self.noise, bounds=((0.1, 5),))
        self.noise = res['x']
        best_noise_obj_value = res['fun']
        print('noise:', self.noise)
        print('obj_value: ', best_noise_obj_value)
        
        best_noise_obj_value = np.inf
        for i in range(0, 5):
            res = minimize(Q, np.random.uniform(0.1, 5), bounds=((0.1, 5),))
            noise = res['x']
            obj_value = res['fun']
            if obj_value < best_noise_obj_value:
                self.noise = noise
                best_noise_obj_value = obj_value
            print('noise: ', noise)
            print('obj_value: ', obj_value)
            
        print('final_noise: ', self.noise)
        print('best_noise_obj_value: ', best_noise_obj_value)
        
        print ('[addPreferences] checking pref pairs')
        for i in range(0,self.preferences.shape[1]):
            r = tuple(self.preferences[0,i,:])
            c = tuple(self.preferences[1,i,:])
            if self.Y[x2ind[r]] <= self.Y[x2ind[c]]:
                print ('   FAILED!  %.2f ! > %.2f'  % (self.Y[x2ind[r]], self.Y[x2ind[c]]))
                print ('which preference:', r, c)
                # print '   FAILED!  %.2f ! > %.2f'  % (self.Y[x2ind[r]], self.Y[x2ind[c]])
                # print '    can we fix it?'
                # if there is nothing preferred to this item, bump it up
                '''
                for r1, c1 in self.preferences:
                    if all(tuple(c1)==tuple(r)):
                        break
                    else:
                        self.Y[x2ind[r]] = self.Y[x2ind[c]] + .1
                        # print '    changed Y to %.2f' % self.Y[x2ind[r]]
                '''
                for j in range(0,self.preferences.shape[1]):
                    if r == tuple(self.preferences[1,j,:]):
                        print('remain')
                        break
                else:
                    self.Y[x2ind[r]] = self.Y[x2ind[c]] + .1
                    print('    changed Y to %.2f' % self.Y[x2ind[r]])
        
        # now we can learn the C matrix
        #self.C = eye(len(self.X), dtype=float) * 5
        self.C = zeros((len(self.X), len(self.X)))

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
                        self.C[i,j] += alpha / (2*self.noise) * (pdf**2/cdf**2 + Z * pdf/cdf)
        try:
            self.L = linalg.cholesky(self.R+linalg.inv(self.C))
        except LinAlgError:
            print ('[addPreferences] GP.C matrix is ill-conditioned, adding regularizer delta = 1')
            for i in range(10):
                self.C += eye(len(self.X))
                try:
                    self.L = linalg.cholesky(self.R+linalg.inv(self.C))
                except LinAlgError:
                    print ('[addPreferences] GP.C matrix is ill-conditioned, adding regularizer delta = %d' % (i+2))
                else:
                    break
                
            
                    
    def addObservationPoint(self, X):
        """
        Add a point at which we will observe, but for which we don't have the
        observation yet.  (Used by the gallery selection algorithms.)
        """
        X = array(X, copy=False, dtype=float, ndmin=2)
        
        if self.augR is None:
            self.augR = self.R.copy()
            self.augX = self.X.copy()
        
        r, m = self._computeAugCorrelations(X)
        self.augR = r_[c_[self.augR, m], c_[m.T, r]]
        
        invC = zeros_like(self.augR)
        invC[:self.C.shape[0], :self.C.shape[0]] = linalg.inv(self.C)
        self.augL = linalg.cholesky(self.augR+invC)
        self.augX = r_[self.augX, X]
        
            
    
    def addData(self, X, Y, G=None):
        """
        I have an idea about how to do this... (see notebook).
        """
        raise NotImplementedError("can't (yet) add explicit ratings to preference GP")
        
    def build_preference_ge(self, weights):
        trajectory_name = 'saffa.h5'
        
        # weights                                       term in paper
        lag_weight = 1                                  # weight on e^l
        contour_weight = weights[0,0]                              # weight on e^c
        angular_weight = 1                              # w_phi and w_psi  
        # learn jerk	
        jerk_weight = weights[0,1]                                # w_j on position
        angular_jerk_weight = 1                         # w_j on angles
        theta_input_weight = 1                          # weight on v_i
        # not necessary
        min_time_weight = 1                             # w_end
        end_time_weight = 0                             # w_len
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
        err_new = np.absolute(end_time -25) 
        c_w_new = np.asscalar(contour_weight)
        j_w_new = np.asscalar(jerk_weight)
        
        # create kml-file and save to directory "kmls"
        generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name, '_new')
        
        c_w_min, j_w_min = self.X[np.argmax(self.Y)]
        contour_weight = c_w_min
        jerk_weight = j_w_min
        options["contour_weight"] = contour_weight
        options["smoothness_weight"] = jerk_weight
        trajGen = QuadCameraContourTrajectory(keyframes, keyorientations, options)
        trajectory, keytimes = trajGen.generate_trajectory()
        end_time = trajectory[trajGen.endTimeStateIndex, -1];
        err_min = np.absolute(end_time -25)
        # create kml-file and save to directory "kmls"
        generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name, '_premax')
        
        print(weights, 'new_sample error is: '+str(err_new))
        print(self.X[np.argmax(self.Y)], 'premax_sample error is: '+str(err_min))
        #pref_decision = input(":(n for new setting parameters and p for pre_max) ")
        pref = np.zeros((2,1,2))
        
        if (err_new < err_min):
            pref[0,0,:] = np.array([c_w_new, j_w_new]).reshape((1,-1))
            pref[1,0,:] = np.array([c_w_min, j_w_min]).reshape((1,-1))
        elif (err_new > err_min):
            pref[0,0,:] = np.array([c_w_min, j_w_min]).reshape((1,-1))
            pref[1,0,:] = np.array([c_w_new, j_w_new]).reshape((1,-1))
        else:       #no preference
            print('finishes')
            return np.array([])
        '''
        if (pref_decision == 'n'):
            pref[0,0,:] = np.array([c_w_new, j_w_new]).reshape((1,-1))
            pref[1,0,:] = np.array([c_w_min, j_w_min]).reshape((1,-1))
        elif (pref_decision == 'p'):
            pref[0,0,:] = np.array([c_w_min, j_w_min]).reshape((1,-1))
            pref[1,0,:] = np.array([c_w_new, j_w_new]).reshape((1,-1))
        else:       #no preference
            print('finishes')
            return np.array([])
        '''
        self.loss.append(err_new)
        
        return pref
        
def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory        
