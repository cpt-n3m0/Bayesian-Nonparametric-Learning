#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import imageio
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sys

np.set_printoptions(precision=2)

#Hyper-parameters
alpha=2
m0 = 0 # updated based on data
V0 = 1000
sh0 = 2
sc0 = 2


def init(x, random=True):
    if random:
        pi = np.random.dirichlet([2] * K, size=1)[0]
        mus = [np.random.normal(loc=m0, scale=V0, size=1)[0] for k in range(K)]
        Vs = [1/np.random.gamma(shape=sh0, scale=1/sc0, size=1)[0] for k in range(K)]
    else:
        pi = [1/K] * K
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(x.reshape((-1,1)))
        mus = kmeans.cluster_centers_
        Vs = [np.random.gamma(shape=sh0, scale=1/sc0, size=1)[0] for k in range(K)]

    
    z = np.array([np.argwhere(np.random.multinomial(1, [1/K] * K) == 1).ravel()[0] for x_i in x]) #initialize z randomly

    
    return pi, mus, Vs, z


def sample_z(x, mus, Vs, pi):
    std_devs = list(map(math.sqrt, Vs)) 
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    
    for k in range(K):
        p_k[:, k] = pi[k] * norm.pdf(x, mus[k], std_devs[k])

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
            Vs[k] = pis[0]
            continue
        Vs[k] = np.random.beta(a, b)
        pis[k] = Vs[k] * (1 - np.sum(pis[:k]))
    return pis

        

def sample_mu(V_k, ks_mean, N):
        
    new_V= 1/(1/V0 + N/ V_k)
    new_m = new_V * (ks_mean * N/V_k + m0/V0)

    return [np.random.normal(loc=new_m[k], scale=math.sqrt(new_V[k])) for k in range(K)]


def sample_v(m_k, ks_mean, ks_var, N):

    new_shapes = sh0 + N/2
    new_rates = 1/sc0 + N/2 * (ks_var + (ks_mean- m_k)**2)

    lmbda = [np.random.gamma(shape=new_shapes[k], scale=1/new_rates[k], size=1)[0] for k in range(K)]
    
    return 1/np.array(lmbda)

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
    for k in range(K):
        x_k = x[z == k]
        likelihood += np.sum(norm.logpdf(x_k, mus[k], Vs[k]))
    return likelihood

   

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

def DP_gibbs_sampler(x, gt=[], niter=100):
    
    global K
    K = truncate(x, alpha)
    print(K)
    
    pi, mus, Vs, z = init(x)
    sampled_mus = np.zeros((niter, K))
    sampled_Vs = np.zeros((niter, K))
    sampled_pis = np.zeros((niter, K))
    ARI = np.zeros((niter,))
    AMI = np.zeros((niter,))
    loglikelihood = np.zeros((niter,))
    sampled_z = z

    sampled_mus[0] = mus
    sampled_Vs[0] = Vs
    sampled_pis[0] = pi

    for n in range(1, niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end="\r", flush=True )
        

        #-------------------------Sampling start-------------------------------
        if n > 1:
            z = sample_z(x, sampled_mus[n-1], sampled_Vs[n-1], sampled_pis[n-1])



        sampled_pis[n] = sample_pi(z)
        
        
        ks_var = np.zeros((K,))
        ks_mean = np.zeros((K,))
        N = np.zeros((K,))
        for k in range(K):
            ks = x[np.argwhere(z==k).ravel()]
            N[k] = ks.shape[0]
            ks_var[k] = np.var(ks)
            ks_mean[k] = np.mean(ks)
            if math.isnan(ks_mean[k]):
                ks_mean[k] = 0
            if math.isnan(ks_var[k]):
                ks_var[k] = 0

        sampled_mus[n] = sample_mu(sampled_Vs[n-1], ks_mean, N )
        sampled_Vs[n] = sample_v(sampled_mus[n], ks_mean, ks_var, N )
        sampled_z = z
        
        loglikelihood[n] = log_likelihood(x, sampled_mus[n], sampled_Vs[n], z)
        if gt != []:
            ARI[n] = adjusted_rand_score(gt, z)
            AMI[n] = adjusted_mutual_info_score(gt, z)
        
    return sampled_z, loglikelihood, ARI, AMI






def synth_data(original_pi, N):
    global K, alpha, m0
    K = len(original_pi)
    #generate synthetic dataset
    components = []
    original_mus = []
    original_Vs = []
    x = np.zeros((N,))
    y = np.zeros((N,))

    for k in range(K):
        loc = np.random.random() * 200
        scale = np.random.random() * 10
        c = np.random.normal(loc=loc, scale=scale, size=N)
        original_mus.append(loc)
        original_Vs.append(scale)
    for i in range(N):
        z = np.argwhere(np.random.multinomial(1, original_pi) == 1).ravel()[0]
        x[i] = np.random.normal(original_mus[z], original_Vs[z], size=1)[0]
        y[i] = z
        
    
    m0 = np.mean(x)
    return x, y



if __name__ == "__main__":

    #Flags
    PLOT =  "-plot" in sys.argv

    if "-maxiter" in sys.argv:
        niter = int(sys.argv[sys.argv.index("-maxiter") + 1]) 
    else:
        niter=1000

    if "-s" in sys.argv:
        original_pi =  sys.argv[sys.argv.index("-s") + 1]
        print(original_pi)
        original_pi = [float(e) for e in original_pi.split(",")]
        N = int(sys.argv[sys.argv.index("-s") + 2]) 
        x, y = synth_data(original_pi, N)

        


    DP_gibbs_sampler(x, niter=niter)
