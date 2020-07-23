#!/usr/bin/python3


import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from scipy.stats import wishart, multivariate_normal
import math
import imageio
from sklearn.cluster import KMeans
import sys
import directionalstats as ds

#np.set_printoptions(precision=4)


def init(x, random=True):
    pi = np.random.dirichlet(alpha, size=1)[0]
    mus = [ds.rvMF(1,mu0, kappa0)[0] for k in range(K)]
    kappas = [ np.random.gamma(a, 1/b , size=1)[0]  for k in range(K)]

    z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x]) #initialize z randomly

    if DEBUG:
        print("INITIAL PARAMETERS")
        print("pi: {}".format(str(pi)))
        print("mus: {}".format(str(mus)))
        print("Vs: {}".format(str(kappas)))
    return pi, mus, kappas, z


def sample_z(x, mus, kappas, pi):
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    pxi = []
    for k in range(K):
        for x_i in x:
            pxi.append( pi[k] * ds.vMFpdf(x_i, mus[k], kappas[k]))
        p_k[:, k] = pxi
        pxi = []

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


def sample_mu(kappa_k, z, x):
    samples = []
    for k in range(K):
        x_k = x[np.argwhere(z == k).ravel()]
        Vn = kappa0 * mu0 + kappa_k[k] * np.sum(x_k, axis=0)
        posterior_mean = Vn / norm(Vn) 
        posterior_kappa = norm(Vn)
        samples.append(ds.rvMF(1, posterior_mean, posterior_kappa)[0])

    return samples




def sample_kappa(means,z, x):
    samples = []
    for k in range(K):
        x_k = x[np.argwhere(z == k).ravel()]
        p = lambda x: ds.kappa_pdf(x, means[k], mu0, kappa0, a, b, x_k)
    #    print("here)")
    #    x = np.arange(0.1, 100, 0.1)
    #    y = [p(e) for e in x] 
    #    plt.plot(x, y)
    #    print(y)
    #    plt.show()
    #    sys.exit()
        samples.append(ds.slice_sampler(p, 1)[-1])
    return samples

images = []
def show(itern, x, z, mus, kappas):
    for k in range(K):
        x_k = x[np.argwhere(z == k).ravel()]
        plt.scatter(x_k[:, 0], x_k[:, 1], label="K={}".format(k))
    plt.title("iter = {}, means = {}, kappas = {}".format(str(itern), mus, kappas), fontsize=8)
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
    pi, mus, kappas, z = init(x)
    
    sampled_mus = np.zeros((niter, K, P))
    sampled_kappas = np.zeros((niter, K))
    sampled_pis = np.zeros((niter, K))
    sampled_z = np.zeros((niter, len(x)))
    
    sampled_mus[0] = np.array(mus)
    sampled_kappas[0] = kappas
    sampled_pis[0] = pi
    sampled_z[0] = z 
    
    for n in range(1, niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end= "\n" if  DEBUG else "\r", flush=not DEBUG )
        
        if n > 1:
            z = sample_z(x, sampled_mus[n - 1], sampled_kappas[n - 1], sampled_pis[n - 1])

        sampled_pis[n] = sample_pi(z)
        sampled_mus[n] = sample_mu(sampled_kappas[n - 1], z, x )
        sampled_kappas[n] = sample_kappa(sampled_mus[n], z, x )
        sampled_z = z
        
        if PLOT: show(n, x, z, sampled_mus[n], sampled_kappas[n])


    print("")
    print("True parameters")
    print("Mus : " + str(np.array(original_mus)))
    print("kappas : " + str(np.array(original_kappas)))
    print("final samples")
    print("final Mus : " + str(np.array(sampled_mus[-1])))
    print("final kappas : " + str(sampled_kappas[-1]))
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



if "-s" in sys.argv:

    K = int(sys.argv[sys.argv.index("-s") + 1]) 
    N = int(sys.argv[sys.argv.index("-s") + 2]) 
    #generate synthetic dataset
    components = []
    original_mus = []
    original_kappas = []
    for i in range(K):
        m = np.random.multivariate_normal([0] * P, np.identity(P), size=2)
        mean = m[0]/norm(m[0])
        concentration = np.random.random() * 30

        components.append(ds.rvMF(N, mean, concentration))     
        original_mus.append(mean)
        original_kappas.append(np.array(concentration))

    if PLOT:
        for c in components:
            plt.scatter(c[:, 0], c[:, 1], s=np.pi * 3)
        plt.show()
    x = np.concatenate(components)
    

elif "-f" in sys.argv:
    # read data from file
    ds = sys.argv[sys.argv.index("-f") + 1]
    K = int(sys.argv[sys.argv.index("-f") + 2]) 
    x = np.genfromtxt(ds, delimiter=',')


np.random.shuffle(x)

#Hyper-parameters
alpha = [2] * K
mu0, _ = ds.circ_mean(x)
#kappa0 = ds.concentration(x)
kappa0 = 0.01
a = 1
b = 0.001


gibbs_sampler(x, niter=niter)
if PLOT :
    imageio.mimsave("convergence_animation.gif", images, duration=0.1)
