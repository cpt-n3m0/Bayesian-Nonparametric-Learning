#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import sys


VERBOSE = False

if "-v" in sys.argv :
    VERBOSE = True


#generate synthetic dataset
x = np.concatenate([np.random.normal(loc=-60, scale=4, size=50), np.random.normal(loc=30, scale=2, size=50), np.random.normal(loc=50, scale=2, size=50)]).ravel()
np.random.shuffle(x)

if VERBOSE:
    print(x)
    plt.hist(x)
    plt.show()


# Sampler
#Hyper-parameters
K = 3
alpha = [2] * K
m0 = 0
V0 = 1
sh0 = 1
sc0 = 2

pi = np.random.dirichlet(alpha, size=1)[0]
mus = [np.random.normal(loc=m0, scale=V0, size=1)[0] for k in range(K)]
mus = [-30, 20, 40]
#Vs = [2, 2, 2]
Vs = [np.random.gamma(shape=sh0, scale=1/sc0, size=1)[0] for k in range(K)]
#pi=[0.5, 0.5]


def sample_z(x, mus, Vs, pi):
    z = []
    for x_i in x :
        p_k = np.zeros((K,), dtype="float64")
        for k in range(K):
            p_k[k] = pi[k] * norm.pdf(x_i, mus[k], math.sqrt(Vs[k]))
            
        p_k = p_k/np.sum(p_k, dtype="float64")
        z.append(np.argwhere(np.random.multinomial(1, p_k) == 1).ravel()[0])
    z = np.array(z)

    plt.hist(x[np.argwhere(z == 0).ravel()], label="0")
    plt.hist(x[np.argwhere(z == 1).ravel()], label="1")
    plt.hist(x[np.argwhere(z == 2).ravel()], label="2")
    plt.show()
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
    return np.random.normal(loc=new_m, scale=math.sqrt(new_V))

def sample_v(m_k, k , z, x):
    global sh0, sc0
    print("sampling V ...")
    x_k = x[np.argwhere(z==k).ravel()] 
    N_k = x_k.shape[0]
    #empirical_mean = np.sum(x_k)/N_k
    #sample_var = np.sum(np.array([(x_i - m_k) ** 2 for x_i in x_k])) / N_k


    new_shape = sh0 + N_k/2
    new_rate = 1/sc0 + 1/2 * np.sum(np.array([(x_i - m_k) ** 2 for x_i in x_k]))
    #new_rate = 1/sc0 + 1/2 * (N_k * sample_var + N_k * (empirical_mean - m_k) ** 2)
    return 1/np.random.gamma(shape=new_shape, scale=1/new_rate, size=1)[0]

def gibbs_sampler(x, pi, mus, Vs, niter=1000):
    
    sampled_mus = mus
    sampled_Vs = Vs 
    sampled_pis = pi
    sampled_z = []
    k = 0 
    for n in range(niter):
        if k == K: k = 0
        print("\nITER " + str(n) + ", K = " + str(k ))
        print("mus : " + str(sampled_mus))
        print("std_devs : " + str([math.sqrt(i) for i in sampled_Vs]))
        print("pis : " + str(sampled_pis))
        z = sample_z(x, sampled_mus, sampled_Vs, sampled_pis)

        ipi = []

        ipi = sample_pi(z)
        #for k in range(K):
        imu = sample_mu(sampled_Vs[k], k, z, x )
        iV = sample_v(imu,k, z, x )

        sampled_z = z
        sampled_mus[k] = imu
        sampled_Vs[k] = iV
        sampled_pis = ipi
        k += 1

    print(sampled_mus)
    print(sampled_Vs)
    print(sampled_pis)


gibbs_sampler(x, pi, mus, Vs)
