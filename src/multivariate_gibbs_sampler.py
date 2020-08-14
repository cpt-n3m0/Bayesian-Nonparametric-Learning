#!/usr/bin/python3


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import invwishart, wishart, multivariate_normal
import math
import imageio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sys


#np.set_printoptions(precision=4)
alpha = 1
m0 = []
cov0 = []
S0 = []
v0 = 0



def init(x, random=True):
    pi = [1/K] * K
    mus = [np.random.multivariate_normal(m0, cov0, size=1) for k in range(K)]
    covs = [wishart.rvs(df=P, scale=(np.identity(P))) for k in range(K)]

    z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x]) #initialize z randomly
    return pi, mus, covs, z


def sample_z(x, mus, sigmas, pi):
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    
    for k in range(K):
        p_k[:, k] = np.log(pi[k]) + multivariate_normal.logpdf(x, mus[k], sigmas[k], allow_singular=True)

    p_k = np.exp(p_k)
    p_k = p_k/np.sum(p_k, axis = 1, dtype="float64")[:, np.newaxis]

    z = np.array(list(map(lambda pv : np.argwhere(np.random.multinomial(1, pv) == 1).ravel()[0], p_k)))
    return np.array(z)
 
def sample_pi(z):
    z_counts=np.zeros(K)
    for k in range(K):
        z_counts[k] = np.where(z == k)[0].shape[0]

    new_alpha = alpha + z_counts
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
        samples.append(invwishart.rvs(df=v_k, scale=S_k))
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


def log_likelihood(x, mus, sigmas, z):
    likelihood = 0.0
    for k in range(K):
        x_k = x[z == k]
        likelihood += np.sum(multivariate_normal.logpdf(x_k, mus[k], sigmas[k], allow_singular=True))
    return likelihood


def mv_gibbs_sampler(x, gt=[], niter=100):
    
    global ITER
    pi, mus, sigmas, z = init(x)
    sampled_mus = np.zeros((niter, K, P))
    sampled_sigmas = np.zeros((niter, K, P, P))
    sampled_pis = np.zeros((niter, K))
    loglikelihood = np.zeros((niter,))
    ARI = np.zeros((niter,))
    AMI = np.zeros((niter,))
    sampled_z = z
    
    sampled_mus[0] = mus
    sampled_sigmas[0] = sigmas
    sampled_pis[0] = pi

    for n in range(1, niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end="\r", flush=True )    

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
        loglikelihood[n] = log_likelihood(x, sampled_mus[n], sampled_sigmas[n], z)
        if gt != []:
            ARI[n] = adjusted_rand_score(gt, z)
            AMI[n] = adjusted_mutual_info_score(gt, z)
        


    if "-mnist" in sys.argv:
        u,  c = np.unique(sampled_z, return_counts=True)
        disp_order = u[np.argsort(c)[::-1]]
        plt.bar(disp_order, np.sort(c)[::-1])
        for k in range(K):
            plt.imshow(np.mean(ox[sampled_z == k], axis=0))
            plt.show()
        plt.plot(np.arange(niter), ARI)
        plt.xlabel("iterations")
        plt.ylabel("ARI")
        plt.show()
        plt.xlabel("iterations")
        plt.ylabel("AMI")
        plt.plot(np.arange(niter), ARI)
        plt.show()

    return sampled_z, loglikelihood, ARI, AMI


def synth_data(original_pi, N, p):
    global P, K, m0, cov0, S0, v0, alpha

    K = len(original_pi)
    P = p
    #generate synthetic dataset
    original_mus = []
    original_covs = []
    x = np.zeros((N,P))
    y = np.zeros((N,))

    for k in range(K):
        mean = [np.random.random() * 100 for j in range(P)]
        cov = wishart.rvs(df=P, scale=(np.identity(P)))
        original_mus.append(mean)
        original_covs.append(np.array(cov))
    for i in range(N):
        z = np.argwhere(np.random.multinomial(1, original_pi) == 1).ravel()[0]
        x[i] =np.random.multivariate_normal(mean=original_mus[z], cov=original_covs[z], size=1)[0] 
        y[i] = z

    alpha = [2] * K
    m0 = np.mean(x, axis=0)
    cov0 = np.matrix(1000 * np.identity(P))
    S0 = np.matrix(np.identity(P))
    v0 = P
    
    return x, y



## interface


#Flags

if __name__ == "__main__":
    PLOT =  "-plot"  in sys.argv

    if "-maxiter" in sys.argv:
        niter = int(sys.argv[sys.argv.index("-maxiter") + 1]) 
    else:
        niter=1000



    if "-s" in sys.argv:
        try:
            p = int(sys.argv[1]) 
        except:
            print("Usage multivariate_gibbs_sampler.py Dimensions [...]")
            sys.exit(0)


        original_pi = sys.argv[sys.argv.index("-s") + 1]
        original_pi = [float(e) for e in original_pi.split(",")]

        N = int(sys.argv[sys.argv.index("-s") + 2])
        x, y = synth_data(original_pi, N, p)
        

    elif "-mnist" in sys.argv:
        # read data from file
        K = 4
        (x, y), (_, _) = mnist.load_data()
        x = x[y < 4]
        y = y[y < 4]
        ox = x[:1000]
        x = x[:1000]
        print(np.unique(y[:1000], return_counts=True))
      #  h = lambda x: hog(x, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1))
     #   x= np.array(list(map(h, x)))
     #   u, c= np.unique(y, return_counts=True)
        h = lambda x: x.ravel()
        x= np.array(list(map(h, x)))

        x = StandardScaler().fit_transform(x)
        p = PCA(n_components = 10)
        x = p.fit_transform(x)
        P = x.shape[1]
        m0 = np.mean(x, axis=0)
        cov0 = np.matrix(1000 * np.identity(P))
        S0 = np.matrix(np.identity(P))
        v0 = P




    mv_gibbs_sampler(x, niter=niter)
    if PLOT :
        imageio.mimsave("convergence_anim.gif", images, duration=0.1)
        plt.plot(list(range(len(ilmeans))), ilmeans)
        plt.savefig("figures/avg_log_likelihood_{}".format(str(M)))
