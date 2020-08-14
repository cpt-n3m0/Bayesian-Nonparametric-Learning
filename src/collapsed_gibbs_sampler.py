#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import math
import imageio
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sys

np.set_printoptions(precision=2)


        
 
images = []
def show(itern, x, z):
    for k in range(K):
        plt.hist(x[np.argwhere(z == k).ravel()], label="K={}".format(k))
    plt.savefig("figures/iter_" + str(itern))
    plt.clf()
    images.append(imageio.imread("figures/iter_"+  str(itern) + ".png"))




def pxi(x_i, x_k, N_k):
    if N_k == 0:
        xm = 0
    else : 
        xm = np.mean(x_k, axis=0)
    m_n = (k0 * m0 + N_k * xm)/(k0 + N_k)
    k_n = k0 + N_k
    a_n = a0 + N_k/2
    b_n = b0 + 0.5 * np.sum((x_k - xm) ** 2 , axis=0) + (k0 * N_k * (xm - m0) ** 2) / (2 * (k0 + N_k))

    df = 2 * a_n
    t_scale = (b_n * (1 + k_n))/(a_n * k_n)

    return t.logpdf(x_i, df, loc=m_n, scale=t_scale)

def collapsed_gibbs_sampler(x, niter=100):
    xlen = x.shape[0]
    z = np.array([np.argwhere(np.random.multinomial(1,[1/K] * K) == 1).ravel()[0] for e in range(xlen)])
    memberships = [x[np.argwhere(z == k).ravel()] for k in range(K)]
    
    for n in range(niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end= "\n" if  DEBUG else "\r", flush=not DEBUG )
        for i in range(xlen):
            p_z = np.zeros((K, ))
            memberships[z[i]] = np.delete(memberships[z[i]], np.where(memberships[z[i]] == x[i])[0])
            for k in range(K):
                N_k = memberships[z[i]].shape[0] 
                p_xi = pxi(x[i] , memberships[z[i]], N_k)
                p_z[k] = math.exp( math.log((N_k + alpha[k]/K)/ (xlen + alpha[k] - 1)) + p_xi  )
            p_z = p_z/np.sum(p_z)
            z[i] =  np.argwhere(np.random.multinomial(1, p_z) == 1).ravel()[0]
            memberships[z[i]] = np.append(memberships[z[i]], x[i]) 
        if PLOT: show(n, x, z)

    x_ks = [x[np.argwhere(z == k).ravel()] for k in range(K)]
    mus = np.array([np.mean(x_k) for x_k in x_ks])
    vs = np.array([np.std(x_k) for x_k in x_ks])
    print("Results : ")
    print("mus : " + str(mus))
    print("vs : " + str(vs))

    print("True:")
    print("mus : " + str(original_mus))
    print("vs : " + str(original_Vs))


                
    
        







## interface


#Flags
DEBUG = False
EARLYSTOP = "--earlystop" in sys.argv
PLOT =  "--no-plot" not in sys.argv

if "-maxiter" in sys.argv:
    niter = int(sys.argv[sys.argv.index("-maxiter") + 1]) 
else:
    niter=1000





if "-s" in sys.argv:
    K = int(sys.argv[sys.argv.index("-s") + 1]) 
    N = int(sys.argv[sys.argv.index("-s") + 2]) 
    #generate synthetic dataset
    components = []
    original_mus = []
    original_Vs = []
    for i in range(K):
        loc = np.random.random() * 100
        scale = np.random.random() * 5
        c = np.random.normal(loc=loc, scale=scale, size=N)
        original_mus.append(loc)
        original_Vs.append(scale)
        components.append(c)
    x = np.concatenate(components)

elif "-f" in sys.argv:
    # read data from file
    ds = sys.argv[sys.argv.index("-f") + 1]
    K = int(sys.argv[sys.argv.index("-f") + 2]) 
    x = np.genfromtxt(ds, delimiter=',')


np.random.shuffle(x)

#Hyper-parameters
alpha = [2] * K
m0 = np.mean(x)
k0 = 1000
a0 = 2
b0 = 2




collapsed_gibbs_sampler(x, niter=niter)
if PLOT :
    imageio.mimsave("convergence_anim.gif", images, duration=0.1)
