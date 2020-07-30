#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import imageio
from sklearn.cluster import KMeans
import sys

np.set_printoptions(precision=2)


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

    if DEBUG:
        print("INITIAL PARAMETERS")
        print("pi: {}".format(str(pi)))
        print("mus: {}".format(str(mus)))
        print("Vs: {}".format(str(Vs)))
    return pi, mus, Vs, z


def sample_z(x, mus, Vs, pi):
    std_devs = list(map(math.sqrt, Vs)) 
    z = []
    p_k = np.zeros((x.shape[0], K), dtype="float64")
    
    for k in range(K):
        p_k[:, k] = pi[k] * norm.pdf(x, mus[k], std_devs[k])

    p_k = p_k/np.sum(p_k, axis = 1, dtype="float64")[:, np.newaxis]

    try:
        z = np.array(list(map(lambda pv : np.argwhere(np.random.multinomial(1, pv) == 1).ravel()[0], p_k)))
    except ValueError:
        print(mus)
        print(Vs)
        sys.exit()
    return np.array(z)
    return z
 
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

        

def sample_mu(V_k, ks_mean, N):
        
    new_V= 1/(1/V0 + N/ V_k)
    new_m = new_V * (ks_mean * N/V_k + m0/V0)

    return [np.random.normal(loc=new_m[k], scale=math.sqrt(new_V[k])) for k in range(K)]


def sample_v(m_k, ks_mean, ks_var, N):

    new_shapes = sh0 + N/2
    new_rates = 1/sc0 + N/2 * (ks_var + (ks_mean- m_k)**2)

    lmbda = [np.random.gamma(shape=new_shapes[k], scale=1/new_rates[k], size=1)[0] for k in range(K)]
    if DEBUG:
        print("sampling V ...")
        print("new_shape: " + str(new_shape))
        print("new_rate: " + str(new_rate))
        print("sampled_precision: " + str(lmbda))


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

def DP_gibbs_sampler(x, niter=100):
    
    global K
    K = truncate(x, alpha)
    print(K)
    pi, mus, Vs, z = init(x)
    sampled_mus = [mus]
    sampled_Vs = [Vs]
    sampled_pis = [pi]
    sampled_z = z
    for n in range(niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end= "\n" if  DEBUG else "\r", flush=not DEBUG )
        
        if DEBUG: 
            print("mus : " + str(sampled_mus[-1]))
            print("std_devs : " + str([math.sqrt(i) for i in sampled_Vs[-1]]))
            print("pis : " + str(sampled_pis[-1]))

        #-------------------------Sampling start-------------------------------
        if n > 0:
            z = sample_z(x, sampled_mus[-1], sampled_Vs[-1], sampled_pis[-1])

        if DEBUG: print(z)

        ipi  = sample_pi(z)
        
        imus = []
        iVs  = []
        
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

        sampled_mus.append(sample_mu(sampled_Vs[-1], ks_mean, N ))
        sampled_Vs.append(sample_v(sampled_mus[-1], ks_mean, ks_var, N ))
        sampled_pis.append(ipi)
        sampled_z = z
        
        if PLOT:
            show(n, x, z,  np.array(imus), np.array(iVs), np.array(ipi))
        if EARLYSTOP and check_stop(x, sampled_mus[-1], sampled_Vs[-1], sampled_z):
            print("")
            print("Convergence detected. early stopping")
            break

    u,  c = np.unique(sampled_z, return_counts=True)

    disp_order = u[np.argsort(c)[::-1]]
    print(disp_order)
    print("")
    print("True parameters")
    print("Mus : " + str(np.array(original_mus)))
    print("std devs : " + str(np.array(original_Vs)))
    print("final samples")
    print("final Mus : " + str(np.array(sampled_mus[-1])[disp_order]))
    print("final std devs : " + str(np.array(list(map(math.sqrt, sampled_Vs[-1][disp_order])))))
    print("final pis: " + str(sampled_pis[-1][disp_order]))
    plt.bar(u, c)
    plt.show()
    






## interface


#Flags

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

if "-K" in sys.argv:
    K = int(sys.argv[sys.argv.index("-K") + 1]) 

np.random.shuffle(x)

#Hyper-parameters
alpha = 1
m0 = np.mean(x)
V0 = 1000
sh0 = 2
sc0 = 2


#x2a = np.arange(1, 1000)
#y2a = [alpha * math.log(e) for e in x2a]
#
#plt.plot(x2a, y2a)
#plt.show()
#



DP_gibbs_sampler(x, niter=niter)
if PLOT :
    imageio.mimsave("convergence_anim.gif", images, duration=0.1)
    plt.plot(list(range(len(ilmeans))), ilmeans)
    plt.savefig("figures/avg_log_likelihood_{}".format(str(M)))
