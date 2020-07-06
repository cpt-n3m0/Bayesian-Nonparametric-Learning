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
        pi = np.random.dirichlet(alpha, size=1)[0]
        mus = [np.random.normal(loc=m0, scale=V0, size=1)[0] for k in range(K)]
        Vs = [np.random.gamma(shape=sh0, scale=1/sc0, size=1)[0] for k in range(K)]
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
    z = []
    for x_i in x :
        p_k = np.zeros((K,), dtype="float64")
        for k in range(K):
            p_k[k] = pi[k] * norm.pdf(x_i, mus[k], math.sqrt(Vs[k]))
        p_k = p_k/np.sum(p_k, dtype="float64")
        try:
            z.append(np.argwhere(np.random.multinomial(1, p_k) == 1).ravel()[0])
        except ValueError:
            print(p_k)
            print(x_i)
            print(mus)
            print(Vs)

    return np.array(z)
#def old_sample_z(x, mus, Vs, pi):
#    z = []
#    for x_i in x :
#        p_k = np.zeros((K,), dtype="float64")
#        log_p_k = np.zeros((K,), dtype="float64")
#        for k in range(K):
#            p_k[k] = pi[k] * norm.pdf(x_i, mus[k], math.sqrt(Vs[k]))
#            
#            if p_k[k] < min_val:
#                print("done")
#                print(p_k[k])
#                p_k[k] = min_val
#
#            log_p_k[k] = math.log(p_k[k])
#
#        A = max(log_p_k) 
#        log_marginal = A + math.log(sum([math.exp(e - A) for e in log_p_k]))
#        normalized_log_p_k = log_p_k - log_marginal
#        p_k = np.array(list(map(math.exp, normalized_log_p_k)))
#        z.append(np.argwhere(np.random.multinomial(1, p_k) == 1).ravel()[0])
#
#    z = np.array(z)
#    return z
 
def sample_pi(z):
    z_counts=np.zeros(K)
    for k in range(K):
        z_counts[k] = np.where(z == k)[0].shape[0]

    new_alpha = alpha + z_counts
    if DEBUG:
        print("sampling pi ...")
        print("new alpha: " + str(new_alpha))
    return np.random.dirichlet(new_alpha)


def sample_mu(V_k, k, z, x):
    global V0, m0
    x_k = x[np.argwhere(z==k).ravel()]
    N_k = x_k.shape[0]
    empirical_mean = np.sum(x_k)/N_k
    new_V= 1/(1/V0 + N_k / V_k)
    new_m = new_V * (empirical_mean * N_k/V_k + m0/V0)
    if DEBUG:
        print("sampling mu ...")
        print("num z_{} : ".format(k) + str(N_k))
        print("new_V: " + str(new_V) )
        print("new_m: " + str(new_m) )
        print("empirical mean: " + str(empirical_mean))
    return np.random.normal(loc=new_m, scale=math.sqrt(new_V))

def sample_v(m_k, k , z, x):
    global sh0, sc0
    x_k = x[np.argwhere(z==k).ravel()] 
    N_k = x_k.shape[0]
    new_shape = sh0 + N_k/2
    new_rate = 1/sc0 + 1/2 * np.sum(np.array([(x_i - m_k) ** 2 for x_i in x_k]))
    lmbda = np.random.gamma(shape=new_shape, scale=1/new_rate, size=1)[0] 
    if DEBUG:
        print("sampling V ...")
        print("new_shape: " + str(new_shape))
        print("new_rate: " + str(new_rate))
        print("sampled_precision: " + str(lmbda))
    return 1/lmbda

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

#def full_log_likelihood(x, mus, Vs, z):
#    likelihood = 0.0
#    pi = sample_pi(z)
#    for i in range(len(x)):
#        likelihood += math.log(sum([pi[k] * norm.pdf(x[i], mus[k], Vs[k]) for k in range(K)]))
#    return likelihood

iter_likelihoods = []
ilmeans = []
def check_stop(x, sampled_mus, sampled_Vs, z):
        iter_likelihoods.append(log_likelihood(x, sampled_mus, sampled_Vs, z))
        ilmeans.append(np.mean(iter_likelihoods[-M:]))
        return len(ilmeans) > 2 and abs(ilmeans[-2] - ilmeans[-1]) < 0.1 
    



def gibbs_sampler(x, niter=100):
    
    global ITER
    pi, mus, Vs, z = init(x)
    sampled_mus = [mus]
    sampled_Vs = [Vs]
    sampled_pis = [pi]
    sampled_z = z
    for n in range(niter):
        print("ITER {}/{}".format(str(n), str(niter)) , end= "\n"  if DEBUG else "\r", flush=not DEBUG )
        
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
        for k in range(K):
            imus.append(sample_mu(sampled_Vs[-1][k], k, z, x ))
            iVs.append(sample_v(imus[-1] ,k, z, x ))

        sampled_pis.append(ipi)
        sampled_mus.append(imus)
        sampled_Vs.append(iVs)
        sampled_z = z
        
        if PLOT:
            show(n, x, z,  np.array(imus), np.array(iVs), np.array(ipi))
        if EARLYSTOP and check_stop(x, sampled_mus[-1], sampled_Vs[-1], sampled_z):
            print("")
            print("Convergence detected. early stopping")
            break


    print("")
    print("True parameters")
    print("Mus : " + str(original_mus))
    print("std devs : " + str(original_Vs))
    print("final samples")
    print("final Mus : " + str(np.mean(sampled_mus[-M:], axis=0)))
    print("final std devs : " + str(np.array(list(map(math.sqrt, np.mean(sampled_Vs[-M:], axis=0))))))
    print("final pis: " + str(np.mean(sampled_pis, axis=0)))
    







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


np.random.shuffle(x)

#Hyper-parameters
M = 20 # number of samples considered for likelihood stop calculation
alpha = [2] * K
m0 = np.mean(x)
V0 = 100000
sh0 = 1
sc0 = 1




gibbs_sampler(x, niter=niter)
if PLOT :
    imageio.mimsave("convergence_anim.gif", images, duration=0.1)
    plt.plot(list(range(len(ilmeans))), ilmeans)
    plt.savefig("figures/avg_log_likelihood_{}".format(str(M)))

