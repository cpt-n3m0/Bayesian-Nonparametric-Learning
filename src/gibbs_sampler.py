#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import imageio
from sklearn.cluster import KMeans
import sys

np.set_printoptions(precision=2)

VERBOSE = False

if "-v" in sys.argv :
    VERBOSE = True


#generate synthetic dataset
K = 4
x1 = np.random.normal(loc=-50, scale=4, size=100)
x2 = np.random.normal(loc=-20, scale=2, size=100)
x3 = np.random.normal(loc=0, scale=1, size=100)
x4 = np.random.normal(loc=50, scale=2, size=100)
x = np.concatenate([x1, x2, x3, x4]).ravel()
np.random.shuffle(x)

if VERBOSE:
    print(x)
    plt.hist(x)
    plt.show()


# Sampler
#Hyper-parameters
alpha = [2] * K
m0 = np.mean(x)
V0 = 1000
sh0 = 1
sc0 = 2

def init_parameters(x, random=False):
    if random:
        pi = np.random.dirichlet(alpha, size=1)[0]
        mus = [np.random.normal(loc=m0, scale=V0, size=1)[0] for k in range(K)]
        Vs = [np.random.gamma(shape=sh0, scale=1/sc0, size=1)[0] for k in range(K)]
    else:
        pi = [1/K] * K
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(x.reshape((-1,1)))
        mus = kmeans.cluster_centers_
        Vs = [np.random.gamma(shape=sh0, scale=1/sc0, size=1)[0] for k in range(K)]


    print("INITIAL PARAMETERS")
    print("pi: {}".format(str(pi)))
    print("mus: {}".format(str(mus)))
    print("Vs: {}".format(str(Vs)))
    return pi, mus, Vs

ITER = 0

def sample_z(x, mus, Vs, pi):
    z = []
    for x_i in x :
        p_k = np.zeros((K,), dtype="float64")
        for k in range(K):
            p_k[k] = pi[k] * norm.pdf(x_i, mus[k], math.sqrt(Vs[k]))
            
        p_k = p_k/np.sum(p_k, dtype="float64")
        z.append(np.argwhere(np.random.multinomial(1, p_k) == 1).ravel()[0])
    z = np.array(z)

    #    plt.show()
    return z

def sample_pi(z):
    print("sampling pi ...")
    z_counts=np.zeros(K)
    for k in range(K):
        z_counts[k] = np.where(z == k)[0].shape[0]

    new_alpha = alpha + z_counts
    print("new alpha: " + str(new_alpha))
    return np.random.dirichlet(new_alpha)


def sample_mu(V_k, k, z, x):
    global V0, m0
    print("sampling mu ...")
    x_k = x[np.argwhere(z==k).ravel()]
    N_k = x_k.shape[0]
    print("num z_{} : ".format(k) + str(N_k))
    empirical_mean = np.sum(x_k)/N_k
    new_V= 1/(1/V0 + N_k / V_k)
    new_m = new_V * (empirical_mean * N_k/V_k + m0/V0)
    print("new_V: " + str(new_V) )
    print("new_m: " + str(new_m) )
    print("empirical mean: " + str(empirical_mean))
    return np.random.normal(loc=new_m, scale=math.sqrt(new_V))

def sample_v(m_k, k , z, x):
    global sh0, sc0
    print("sampling V ...")
    x_k = x[np.argwhere(z==k).ravel()] 
    N_k = x_k.shape[0]
    new_shape = sh0 + N_k/2
    new_rate = 1/sc0 + 1/2 * np.sum(np.array([(x_i - m_k) ** 2 for x_i in x_k]))
    print("new_shape: " + str(new_shape))
    print("new_rate: " + str(new_rate))
    lmbda = np.random.gamma(shape=new_shape, scale=1/new_rate, size=1)[0] 
    print("sampled_precision: " + str(lmbda))
    return 1/lmbda

images = []
def show(itern, x, z, mus, vs, pis):
    for k in range(K):
        plt.hist(x[np.argwhere(z == k).ravel()], label="K={}".format(k))
    std_devs = np.array([math.sqrt(i) for i in vs])
    plt.title("iter = {}, means = {}, std_devs = {}, pis = {}".format(str(itern), mus, std_devs, pis), fontsize=8)
    plt.savefig("iter_" + str(itern))
    plt.clf()
    images.append(imageio.imread("iter_"+  str(itern) + ".png"))


def gibbs_sampler(x, niter=100):
    
    global ITER
    pi, mus, Vs = init_parameters(x, random=True)
    sampled_mus = np.array(mus)
    sampled_Vs = np.array(Vs) 
    sampled_pis = np.array(pi)
    sampled_z = []
    for n in range(niter):
        ITER = n
        print("\nITER " + str(n) )
        print("mus : " + str(sampled_mus))
        print("std_devs : " + str([math.sqrt(i) for i in sampled_Vs]))
        print("pis : " + str(sampled_pis))
        if n == 0:
            z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x])
        else:
            z = sample_z(x, sampled_mus, sampled_Vs, sampled_pis)

        print(z)


        ipi = sample_pi(z)
        for k in range(K):
            sampled_mus[k] = sample_mu(sampled_Vs[k], k, z, x )
            sampled_Vs[k] = sample_v(sampled_mus[k],k, z, x )


        sampled_z = z
        sampled_pis = ipi

        show(n, x, z, sampled_mus, sampled_Vs, sampled_pis)
        
    print("final samples")
    print(sampled_mus)
    print([math.sqrt(i) for i in sampled_Vs])
    print(sampled_pis)
    



gibbs_sampler(x)
imageio.mimsave("convergence_anim.gif", images, duration=0.1)
