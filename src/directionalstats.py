
import numpy as np
import scipy as sc
import scipy.stats
from scipy.special  import iv 
import scipy.linalg as la
from scipy.stats import norm, poisson
import math
import matplotlib.pyplot as plt
import sys

# vMF sampling function code taken from https://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python 

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
    b = dim / (np.sqrt(4*kappa*kappa + dim*dim) + 2*kappa)
    x = (1-b) / (1+b)
    c = kappa*x + dim*np.log(1-x*x)

    y = []
    for i in range(0,n):
        done = False
        while not done:
            z = sc.stats.beta.rvs(dim/2,dim/2)
            w = (1 - (1+b)*z) / (1 - (1-b)*z)
            u = sc.stats.uniform.rvs()
            if kappa*w + dim*np.log(1-x*w) - c >= np.log(u):
                done = True
        y.append(w)

    print(y)
    return y


def rvMF(n, theta):
    dim = len(theta)
    kappa = np.linalg.norm(theta)
    mu = theta / kappa
    result = []
    for sample in range(0,n):
        w = rW(n, kappa, dim)
        #v = sample_tangent_unit(mu)

        v = np.random.randn(dim)
        v = v / np.linalg.norm(v) 
        result.append(np.sqrt(1-w**2)*v + w*mu)

    return result
## end of copied code

def vMFpdf(x, mu, kappa, p=1):
    C_p = lambda k: k ** (p/2 - 1)/( (2 * math.PI)**(p/2) * iv(p/2 - 1, k))

    d = C_p(kappa) * np.exp(kappa * mu * x)

def kappa_conditional_pdf(x, mu, mu0, a, b):
    p = mu.shape[0]
    return (x ** (p/2 - 1)/iv(p/2 - 1, x)) * np.exp(b * x * np.dot(mu , mu0.T))

def fix_wbounds(slb, srb, p, y, increment = 0.5):
    # fix right bound
    while  y < p(srb) :
        srb += increment
    while y < p(slb):
        slb -= increment
    return slb, srb
def slice_sampler(p, w, niter=5000):
    default_w = w
    samples = np.zeros((niter,))
    x = 1
    for n in range(niter):
        y = np.random.random() * p(x)
        print("X = " + str(x))
        print(p(x))
        offset = np.random.random() * w
        w_prime = np.random.random() * w
        w_lbound , w_rbound = fix_wbounds(x - offset, x + (w - offset), p, y )
        proposal = w_lbound + np.random.random() * (w_rbound - w_lbound)
        while y > p(proposal):
            proposal = w_lbound + np.random.random() * (w_rbound - w_lbound)

        samples[n] = proposal
        x = proposal
        w = default_w


    print(np.mean(samples))
    print(math.sqrt(np.var(samples)))
    return samples



if __name__ == "__main__" :
    mu0 = np.array([0.15384193,0.14248371,0.9777684 ])
    mu = np.array([0.84606022, 0.42302811, 0.32439069] )

 #   m = np.random.multivariate_normal([1, 1, 3], [[5, 0, 0], [0, 5, 0], [0, 0,  5]], size=2)
 #   mu = m[0]/np.linalg.norm(m[0])
 #   print(mu)
 #   mu0 = m[1]/np.linalg.norm(m[1])
 #   print(mu0)
    p = lambda x: kappa_conditional_pdf(x, mu , mu0, 1, 2)
    plt.hist(slice_sampler(p, 1, niter=10000))
    plt.show()

