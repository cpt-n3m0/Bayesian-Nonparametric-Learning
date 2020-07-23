
import numpy as np
from numpy.linalg import norm
import scipy as sc
import scipy.stats
from scipy.special  import iv 
import scipy.linalg as la
from scipy.stats import  gamma
import math
from math import log
import matplotlib.pyplot as plt
import sys

# vMF sampling function code taken ( with substantial corrections and modifications)) from https://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python 

maxkappa = 300


def sample_tangent_unit(mu):
    mat = np.matrix(mu)

    if mat.shape[1]>mat.shape[0]:
        mat = mat.T

    U,_,_ = la.svd(mat)
    nu = np.matrix(np.random.randn(mat.shape[0])).T
    x = np.dot(U[:,1:],nu[1:,:])
    return x/la.norm(x)

def rW(n, kappa, m):
    dim = m-1
    b =(np.sqrt(4 * kappa**2 + dim**2) - 2 * kappa) /  dim  
    x = (1-b) / (1+b)
    c = kappa*x + dim*np.log(1-x**2)

    done = False
    while not done:
        z = sc.stats.beta.rvs(dim/2,dim/2)
        w = (1 - (1+b)*z) / (1 - (1-b)*z)
        u = sc.stats.uniform.rvs()
        if kappa*w + dim*np.log(1-x*w) - c >= np.log(u):
            done = True

    
    return w


def rvMF(n, mu, kappa):
    dim = len(mu)
    samples = np.zeros((n,dim))
    for s in range(0,n):
        w = rW(n, kappa, dim)
        v = sample_tangent_unit(mu).ravel()
        #v = np.random.randn(dim)
        #v = v / np.linalg.norm(v) 
        samples[s] = np.sqrt(1-w**2)*v + w*mu

    return samples
## end of copied code

def vMFpdf(x, mu, kappa):
    p = len(mu)
    C_p = lambda k: (k ** (p/2 - 1))/( ((2 * math.pi)**(p/2)) * iv(p/2 - 1, k))

    d = C_p(kappa) * np.exp(kappa * np.dot(mu , x.T))
    return d

def vMFlogpdf(x, mu, kappa):
    p = len(mu)
    logC_p = lambda k: (p/2 - 1)*log(k) - p/2 * log(2 * math.pi) - log( iv(p/2 - 1, k))

    d = logC_p(kappa) + kappa * np.dot(mu , x.T)
    return d



def kappa_pdf(x, mu, mu0, k0, a, b, data):
    if x < 0 or x > maxkappa:
        return 0
    p = mu.shape[0]

    logprior = gamma.logpdf(x, a, scale=1/b)

    loglikelihood = np.sum([vMFlogpdf(e, mu, x) for e in data ])

    return math.exp(loglikelihood + logprior)


    #return (x ** (p/2 - 1)/iv(p/2 - 1, x)) ** a  * np.exp(b * x * np.dot(mu , mu0.T))


    
def circ_mean(x):
    x_bar = np.mean(x, axis=0) 
    R_bar = la.norm(x_bar)
    return x_bar/R_bar, R_bar


def concentration(x):
    _, R_bar = circ_mean(x)
    kappa = (R_bar* (len(x[0]) - R_bar ** 2)) / (1 - R_bar ** 2)
    return kappa

def wstep_out(slb, srb, p, y, increment = 0.5):
    # fix right bound
    while not math.isinf(p(srb)) and  y < p(srb) :
        srb += increment
    
    if slb < 0: 
        slb=0
    else:
        while not math.isinf(p(slb)) and y < p(slb):
            slb -= increment
            if slb < 0:
                slb = 0
                break

    return slb, srb
def slice_sampler(p, w, niter=5, step_out=True):
    default_w = w
    samples = np.zeros((niter,))
    x = 1
    for n in range(niter):
        y = np.random.random() * p(x)
        offset = np.random.random() * w
        w_lbound , w_rbound = x - offset, x + (w - offset)
        if step_out:
            w_lbound, w_rbound = wstep_out(w_lbound, w_rbound, p , y) 
        proposal = w_lbound + np.random.random() * (w_rbound - w_lbound)
        while y > p(proposal):
            proposal = w_lbound + np.random.random() * (w_rbound - w_lbound)

        samples[n] = proposal
        x = proposal
        w = default_w


    return samples



if __name__ == "__main__" :
    #mu0 = np.array([0.15384193,0.14248371,0.9777684 ])
    #mu = np.array([0.84606022, 0.42302811] )
#[0.92297639 0.38485656]
#[ 0.60070391 -0.79947158]
#[ 0.85144824 -0.52443865]
#[0.84784105 0.53025046]
#[0.84917719 0.52810804]
#[-0.89956008  0.43679704]
#[ 0.99151889 -0.12996263]
#[0.07180866 0.99741843]

    vMFpdf(np.array([0,-1]), np.array([0, 1]), 0.5)
    sys.exit()
    m = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=2)
    mu = m[0]/np.linalg.norm(m[0])
    print(mu)
    p = lambda x: kappal_pdf(x, mu , mu, 1, 0.9)
    s = slice_sampler(p, 5)
    
    
    print(s[-1])
    #plt.show()
    vmfs = rvMF(50000, mu, s[-1])
    print("standard sample mean = "  + str(np.mean(vmfs, axis=0)))
    print("estimated mean = " + str(circ_mean(vmfs)))
    print("estimated concentration = " + str(concentration(vmfs)))
    plt.scatter(vmfs[:, 0], vmfs[:, 1])
    plt.show()

