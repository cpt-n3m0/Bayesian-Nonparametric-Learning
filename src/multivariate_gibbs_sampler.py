#!/usr/bin/python3


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import wishart, multivariate_normal
import math
import imageio
from sklearn.cluster import KMeans
import sys

#np.set_printoptions(precision=4)


def init(x, random=True):
    pi = np.random.dirichlet(alpha, size=1)[0]
    mus = [np.random.multivariate_normal(m0, cov0, size=1) for k in range(K)]
    covs = [np.linalg.inv(wishart.rvs(df=P, scale=(np.identity(P)))) for k in range(K)]

    z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x]) #initialize z randomly

    if DEBUG:
        print("INITIAL PARAMETERS")
        print("pi: {}".format(str(pi)))
        print("mus: {}".format(str(mus)))
        print("Vs: {}".format(str(Vs)))
    return pi, mus, covs, z


def sample_z(x, mus, sigmas, pi):
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    
    for k in range(K):
        p_k[:, k] = pi[k] * multivariate_normal.pdf(x, mus[k], sigmas[k])

    p_k = p_k/np.sum(p_k, axis = 1, dtype="float64")[:, np.newaxis]

    z = np.array(list(map(lambda pv : np.argwhere(np.random.multinomial(1, pv) == 1).ravel()[0], p_k)))
    return np.array(z)
 
def sample_pi(z):
    z_counts=np.zeros(K)
    for k in range(K):
        z_counts[k] = np.where(z == k)[0].shape[0]

    new_alpha = alpha + z_counts
    if DEBUG:
        print("sampling pi ...")
        print("new alpha: " + str(new_alpha))
    return np.random.dirichlet(new_alpha)


def sample_mu(sig_k, ks_mean, N):
    
    samples = []
    for k in range(K):
        cov_k = inv(cov0) + N[k] * inv(sig_k[k])
        cov_k = inv(cov_k)
        posterior_add = np.matmul(inv(sig_k[k]), (N[k] *  ks_mean[k])) + np.matmul(inv(cov0), m0)[0]
        m_k = np.matmul(cov_k , np.array(posterior_add)[0])
        m_k = np.array(m_k)[0]
        samples.append(np.random.multivariate_normal(m_k, cov_k))

    return samples




def sample_v(means,x, z):
    samples = []
    for k in range(K):
        x_k = x[np.argwhere(z == k).ravel()]
        v_k = v0 + x_k.shape[0]
        S_k = S0 + sum([np.matrix((x_i - means[k])).T @ np.matrix(x_i - means[k]) for x_i in x_k])
        samples.append(inv(wishart.rvs(df=v_k, scale=inv(S_k))))
    return samples

images = []
def show(itern, x, z, mus, vs, pis):
    for k in range(K):
        plt.hist(x[np.argwhere(z == k).ravel()], label="K={}".format(k))
    std_devs = np.array([math.sqrt(i) for i in vs])
    plt.title("iter = {}, means = {}, std_devs = {}, pis = {}".format(str(itern), mus, std_devs, pis), fontsize=8)
    plt.savefig("figures/iter_" + str(itern))
    plt.clf()
    images.append(imageio.imread("figures/iter_"+  str(itern) + ".png"))


def log_likelihood(x, mus, Vs, z):
    likelihood = 0.0
    for i in range(len(x)):
        k = z[i]
        likelihood += math.log(norm.pdf(x[i], mus[k], Vs[k]))
    return likelihood

iter_likelihoods = []
ilmeans = []
def check_stop(x, sampled_mus, sampled_Vs, z):
        iter_likelihoods.append(log_likelihood(x, sampled_mus, sampled_Vs, z))
        ilmeans.append(np.mean(iter_likelihoods[-M:]))
        return len(ilmeans) > 2 and abs(ilmeans[-2] - ilmeans[-1]) < 0.1 
    



def gibbs_sampler(x, niter=100):
    
    global ITER
    pi, mus, sigmas, z = init(x)
    sampled_mus = [mus]
    sampled_sigmas = [sigmas]
    sampled_pis = [pi]
    sampled_z = z
    for n in range(niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end= "\n" if  DEBUG else "\r", flush=not DEBUG )
        
        if DEBUG: 
            print("mus : " + str(sampled_mus[-1]))
            print("std_devs : " + str([math.sqrt(i) for i in sampled_simgas[-1]]))
            print("pis : " + str(sampled_pis[-1]))

        #-------------------------Sampling start-------------------------------
        if n > 0:
            z = sample_z(x, sampled_mus[-1], sampled_sigmas[-1], sampled_pis[-1])

        sampled_pis.append(sample_pi(z))
        
        ks_mean = np.zeros((K, P)) 
        N = np.zeros((K,))
        for k in range(K):
            ks = x[np.argwhere(z==k).ravel()]
            N[k] = ks.shape[0]
            if N[k] == 0:
                ks_mean[k] = [0] * P
                continue
            ks_mean[k] = np.mean(ks, axis=0)

        sampled_mus.append(sample_mu(sampled_sigmas[-1], ks_mean, N ))
        sampled_sigmas.append(sample_v(sampled_mus[-1],x, z ))
        sampled_z = z
        
        if PLOT:
            show(n, x, z,  np.array(imus), np.array(iVs), np.array(ipi))
        if EARLYSTOP and check_stop(x, sampled_mus[-1], sampled_Vs[-1], sampled_z):
            print("")
            print("Convergence detected. early stopping")
            break


    print("")
    print("True parameters")
    print("Mus : " + str(np.array(original_mus)))
    print("std devs : " + str(np.array(original_covs)))
    print("final samples")
    print("final Mus : " + str(np.array(sampled_mus[-1])))
    print("final std devs : " + str(sampled_sigmas[-1]))
    print("final pis: " + str(sampled_pis[-1]))
    







## interface


#Flags
try:
    P = int(sys.argv[1]) 
except:
    print("Usage multivariate_gibbs_sampler.py Dimensions [...]")
    sys.exit(0)

MARGINALS = False

DEBUG = "-d" in sys.argv
EARLYSTOP = "--earlystop" in sys.argv
PLOT =  "--no-plot" not in sys.argv

if "-maxiter" in sys.argv:
    niter = int(sys.argv[sys.argv.index("-maxiter") + 1]) 
else:
    niter=1000



if "-getmarginals" in sys.argv:
    MARGINALS = True


if "-s" in sys.argv:

    K = int(sys.argv[sys.argv.index("-s") + 1]) 
    N = int(sys.argv[sys.argv.index("-s") + 2]) 
    #generate synthetic dataset
    components = []
    original_mus = []
    original_covs = []
    for i in range(K):
        mean = [np.random.random() * 50 for j in range(P)]
        cov = np.linalg.inv(wishart.rvs(df=P, scale=(np.identity(P))))
        components.append(np.random.multivariate_normal(mean=mean, cov=cov, size=N))     
        original_mus.append(mean)
        original_covs.append(np.array(cov))

    x = np.concatenate(components)
    if PLOT:
        for c in components:
            plt.scatter(x[:, 0], x[:, 1], s=np.pi * 3)
        plt.show()
        sys.exit(-1)
    

elif "-f" in sys.argv:
    # read data from file
    ds = sys.argv[sys.argv.index("-f") + 1]
    K = int(sys.argv[sys.argv.index("-f") + 2]) 
    x = np.genfromtxt(ds, delimiter=',')


np.random.shuffle(x)

#Hyper-parameters
M = 20 # number of samples considered for likelihood stop calculation
alpha = [2] * K
m0 = np.mean(x, axis=0)
cov0 = np.matrix(np.diag(np.diag(np.zeros((P, P)) + 1000)))
S0 = np.identity(P)
v0 = P



gibbs_sampler(x, niter=niter)
if PLOT :
    imageio.mimsave("convergence_anim.gif", images, duration=0.1)
    plt.plot(list(range(len(ilmeans))), ilmeans)
    plt.savefig("figures/avg_log_likelihood_{}".format(str(M)))
