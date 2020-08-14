#!/usr/bin/python3


import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from scipy.stats import wishart, multivariate_normal
import math
import imageio
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sys
import directionalstats as ds
import time
import nlp
#np.set_printoptions(precision=4)

alpha = []
mu0 = []
kappa0 = 0.001
a = 1
b = 0.1
K =0
P =0


def init(x, random=True):
    pi = np.random.dirichlet(alpha, size=1)[0]
    mus = [ds.rvMF(1,mu0, kappa0)[0] for k in range(K)]
    kappas = [ np.random.random() * 300  for k in range(K)]

    z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x]) #initialize z randomly


    return pi, mus, kappas, z


def sample_z(x, mus, kappas, pi):
    start = time.time()
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    pxi = []
    for k in range(K):
        p_k[:, k] =  np.exp(math.log(pi[k]) + ds.vMFlogpdf(x, mus[k], kappas[k]))

    p_k = p_k/np.sum(p_k, axis = 1, dtype="float64")[:, np.newaxis]

    z = np.array(list(map(lambda pv : np.argwhere(np.random.multinomial(1, pv) == 1).ravel()[0], p_k)))

    return np.array(z)
 
def sample_pi(z):
    z_counts=np.zeros(K)
    for k in range(K):
        z_counts[k] = np.where(z == k)[0].shape[0]

    new_alpha = alpha + z_counts
    
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


sliced = 0


def sample_kappa(means,z, x, start):
    samples = []
    for k in range(K):
        x_k = x[np.argwhere(z == k).ravel()]
        p = lambda x: ds.kappa_pdf(x, means[k], mu0, kappa0, a, b, x_k)
      #  if x_k.shape[0] == 0:
      #      print("here)")
      #      x = np.arange(0.1, 100, 0.1)
      #      y = [p(e) for e in x] 
      #      plt.plot(x, y)
      #      print(y)
      #      plt.show()
      #      sys.exit()
        samples.append(ds.slice_sampler(start[k], p, 3, niter=2)[-1])
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


def log_likelihood(x, mus, kappas, z):
    likelihood = 0.0
    for k in range(K):
        x_k = x[z == k]
        likelihood += np.sum(ds.vMFlogpdf(x_k, mus[k], kappas[k]))
    return likelihood

   



def vmf_gibbs_sampler(x, gt=[], niter=100):
    pi, mus, kappas, z = init(x)
    
    sampled_mus = np.zeros((niter, K, P))
    sampled_kappas = np.zeros((niter, K))
    sampled_pis = np.zeros((niter, K))
    loglikelihood = np.zeros((niter,))
    ARI = np.zeros((niter,))
    AMI = np.zeros((niter,))
    
    sampled_mus[0] = np.array(mus)
    sampled_kappas[0] = kappas
    sampled_pis[0] = pi
    
    for n in range(1, niter):
        print("ITER {}/{}".format(str(n), str(niter)) )
        
        if n > 1:
            z = sample_z(x, sampled_mus[n - 1], sampled_kappas[n - 1], sampled_pis[n - 1])

        sampled_pis[n] = sample_pi(z)
        sampled_mus[n] = sample_mu(sampled_kappas[n - 1], z, x )
        sampled_kappas[n] = sample_kappa(sampled_mus[n], z, x , sampled_kappas[n - 1])
        
        loglikelihood[n] = log_likelihood(x, sampled_mus[n], sampled_kappas[n], z)
        if gt != []:
            ARI[n] = adjusted_rand_score(gt, z)
            AMI[n] = adjusted_mutual_info_score(gt, z)
        



    return z, loglikelihood, ARI, AMI
    


def synth_data(original_pi, N, p):
    global K, P, mu0, alpha
    K = len(original_pi)
    P = p
    #generate synthetic dataset
    original_mus = []
    original_kappas = []
    x = np.zeros((N,P))
    y = np.zeros((N,))

    for k in range(K):
        m = np.random.multivariate_normal([0] * P, np.identity(P), size=2)
        mean = m[0]/norm(m[0])
        concentration = np.random.random() * 30
        original_mus.append(mean)
        original_kappas.append(concentration)
    for i in range(N):
        z = np.argwhere(np.random.multinomial(1, original_pi) == 1).ravel()[0]
        x[i] = ds.rvMF(1, original_mus[z], original_kappas[z])[0]
        y[i] = z

    alpha = [2] * K
    mu0, _ = ds.circ_mean(x)
    return x, y



if __name__ == "__main__":
## interface


    PLOT =  "-plot" in sys.argv

    if "-maxiter" in sys.argv:
        niter = int(sys.argv[sys.argv.index("-maxiter") + 1]) 
    else:
        niter=1000



       
    if "-s" in sys.argv:
        try:
            p = int(sys.argv[1]) 
        except:
            print("Usage vMF_gibbs_sampler.py Dimensions [...]")
            sys.exit(0)


        original_pi = sys.argv[sys.argv.index("-s") + 1]
        print(original_pi)
        original_pi = [float(e) for e in original_pi.split(",")]
        N = int(sys.argv[sys.argv.index("-s") + 2])
        x, y = synth_data(original_pi, N, p)
    elif "-nlp":

        x, y, K = nlp.get_data(50)
        P = x.shape[1]
        alpha = [2] * K
        mu0, _ = ds.circ_mean(x)
        print(x.dtype)
        
        

    #Hyper-parameters
    #kappa0 = ds.concentration(x)

    vmf_gibbs_sampler(x, niter=niter)
    if PLOT :
        imageio.mimsave("convergence_animation.gif", images, duration=0.1)
