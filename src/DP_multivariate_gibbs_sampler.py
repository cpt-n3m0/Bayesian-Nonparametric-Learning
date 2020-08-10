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
    mus = [np.random.multivariate_normal(m0, cov0, size=1) for k in range(K)]
    covs = [np.linalg.inv(wishart.rvs(df=P, scale=(np.identity(P)))) for k in range(K)]

    z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x]) #initialize z randomly

    if DEBUG:
        print("INITIAL PARAMETERS")
        print("pi: {}".format(str(pi)))
        print("mus: {}".format(str(mus)))
        print("Vs: {}".format(str(Vs)))
    return  mus, covs, z


def sample_z(x, mus, sigmas, pi):
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    
    for k in range(K):
        try:
            p_k[:, k] = pi[k] * multivariate_normal.pdf(x, mus[k], sigmas[k])
        except:
            print(mus)
            print(sigmas)

    p_k = p_k/np.sum(p_k, axis = 1, dtype="float64")[:, np.newaxis]

    z = np.array(list(map(lambda pv : np.argwhere(np.random.multinomial(1, pv) == 1).ravel()[0], p_k)))
    return np.array(z)
 
def sample_pi(z):
    pis = np.zeros((K,))
    Vs = np.zeros((K,))
    for k in range(K):
        N_k = np.where(z == k)[0].shape[0]
        N_l = np.where(z > k)[0].shape[0]
        a = 1 + N_k
        b = alpha + N_l
        if k == 0:
            pis[0] = np.random.beta(a, b)
            Vs[k] = 1 - pis[0]
            continue
        Vs[k] = np.random.beta(a, b)
        pis[k] = Vs[k] * (1 - np.sum(pis[:k]))
    return pis

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



def truncate(x, alpha):
    mdense = lambda n: 4 * x.shape[0] * math.exp(-(n - 1)/alpha)
    # xa = np.arange(100)
    # ya = [mdense(e) for e in xa]
    # print(ya[20:40])
    max_K = 50
    candidate_K = 2
    while mdense(candidate_K) > 0.5 and candidate_K < max_K:
        candidate_K += 1

    return candidate_K   



def gibbs_sampler(x, niter=100):
    global K
    K = truncate(x, alpha) 
    mus, sigmas, z = init(x)

    sampled_mus = np.zeros((niter, K, P))
    sampled_sigmas = np.zeros((niter, K, P, P))
    sampled_pis = np.zeros((niter, K))
    
    sampled_mus[0] = mus
    sampled_sigmas[0] = sigmas
    sampled_pis[0] = [1/K] * K
    sampled_z = z
    for n in range(1, niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end= "\n" if  DEBUG else "\r", flush=not DEBUG )
        
        #-------------------------Sampling start-------------------------------
        if n > 1:
            z = sample_z(x, sampled_mus[n-1], sampled_sigmas[n-1], sampled_pis[n-1])

        sampled_pis[n] = sample_pi(z)
        
        ks_mean = np.zeros((K, P)) 
        N = np.zeros((K,))
        for k in range(K):
            ks = x[np.argwhere(z==k).ravel()]
            N[k] = ks.shape[0]
            if N[k] == 0:
                ks_mean[k] = [0] * P
                continue
            ks_mean[k] = np.mean(ks, axis=0)


        sampled_mus[n] = sample_mu(sampled_sigmas[n - 1], ks_mean, N )
        sampled_sigmas[n] = sample_v(sampled_mus[n - 1],x, z)
        sampled_z = z

        if PLOT:
            show(n, x, z,  np.array(imus), np.array(iVs), np.array(ipi))

    u,  c = np.unique(sampled_z, return_counts=True)
    disp_order = u[np.argsort(c)[::-1]]

    print("")
    print("True parameters")
    print("Mus : " + str(np.array(original_mus)))
    print("std devs : " + str(np.array(original_covs)))
    print("final samples")
    print("final Mus : " + str(np.array(sampled_mus[-1])[disp_order]))
    print("final std devs : " + str(np.array(sampled_sigmas[-1])[disp_order]))
    print("final pis: " + str(np.array(sampled_pis[-1])[disp_order]))
    
    plt.bar(disp_order, np.sort(c)[::-1])
    plt.show()






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
alpha = 2
m0 = np.mean(x, axis=0)
cov0 = np.matrix(np.diag(np.diag(np.zeros((P, P)) + 1000)))
S0 = np.identity(P)
v0 = P



gibbs_sampler(x, niter=niter)
if PLOT :
    imageio.mimsave("convergence_anim.gif", images, duration=0.1)
    plt.plot(list(range(len(ilmeans))), ilmeans)
    plt.savefig("figures/avg_log_likelihood_{}".format(str(M)))
