The work I have done is initially based on https://github.com/misterwindupbird/IBO (for basic struvture) and ei function is initially based on https://github.com/fmfn/BayesianOptimization/blob/master/bayes_opt/helpers.py. Another useful link I have found is https://github.com/aaronpmishkin/gaussian_processes

The work is in demo.py, pref_GP.py and ei.py. You can run demo.py to reproduce my project for optimizing 2D jerk-contour weights learning.

In demo.py, the main function is demoPrefLearning(), which is the main function for our project. pref_GP contains model and functions for preference learning with gaussian process, and ei.py is about selecting the next new point. There are some comments of functions in each file.

Some parameters for testing are defined at the beginning of demo.py, pref_GP.py and ei.py, e.g., trajectory_name, goal_time and some bounds for hyperparameters and weights.

EI txt is used for recording averaged EI value for each iteration in the test.

end_time_saffa_j_c.py is to generate endtime of discretized parameter space for test and then we use generate_sample_j_c.py to select normalized parameter space based on endtime distribution.



