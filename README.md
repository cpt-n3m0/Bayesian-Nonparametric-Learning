# CS5099

Basic run command : python3 *gibbs_sampler [Dimensions] [-mnist] [-s pi_1,pi_2,..,pi_k Number_of_samples] [-maxiter N] 
-mnist: to run the mnist application (only valid for multivariate_gibbs_sampler.py and DP_multivariate_gibbs_sampler.py )
-s : generate sythentic dataset, requried if not using -mnist takes as argument comma sparated pi vector and total number of samples
-maxiter: specify number of iterations 
Dimensions: if multivariate distributions, required for all but gibbs_sampler.py and DP_gibbs_sampler.py

