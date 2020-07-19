
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


def rvMF(n, mu, kappa):
    dim = len(mu)
    result = []
    for sample in range(0,n):
        w = rW(n, kappa, dim)
        #v = sample_tangent_unit(mu)
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v) 
        result.append(np.sqrt(1-w**2)*v + w*mu)

    return result
## end of copied code

def vMFpdf(x, mu, kappa):
    p = len(mu)
    C_p = lambda k: (k ** (p/2 - 1))/( ((2 * math.PI)**(p/2)) * iv(p/2 - 1, k))

    d = C_p(kappa) * np.exp(kappa * np.dot(mu , x.T))

def kappa_conditional_pdf(x, mu, mu0, a, b):
    p = mu.shape[0]
    return (x ** (p/2 - 1)/iv(p/2 - 1, x)) ** a  * np.exp(b * x * np.dot(mu , mu0.T))

def wstep_out(slb, srb, p, y, increment = 0.5):
    # fix right bound
    while not math.isinf(p(srb)) and  y < p(srb) :
        srb += increment
    
    while math.isinf(p(srb)):
        srb -= increment
    if slb < 0: 
        slb=0
    else:
        while not math.isinf(p(slb)) and y < p(slb):
            print(slb)
            print(p(slb))
            slb -= increment
            if slb < 0:
                slb = 0
                break

    while math.isinf(p(slb)):
        srb += increment

    return slb, srb
def slice_sampler(p, w, niter=500, step_out=True):
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
    #mu = np.array([0.84606022, 0.42302811, 0.32439069] )

    m = np.random.multivariate_normal([1, 3, 5, 1], [[7, 0, 0, 0], [0, 7, 0, 0], [0, 0, 7, 0], [0, 0, 0, 7] ], size=2)
    mu = m[0]/np.linalg.norm(m[0])
    mu0 = m[1]/np.linalg.norm(m[1])
    p = lambda x: kappa_conditional_pdf(x, mu , mu, 1, 0.9)
    x = np.arange(0, 100, 10e-3)
    y = [p(e) for e in x]
    #plt.plot(x,y )
    #plt.show()
    s = slice_sampler(p, 5)
    
    
    #plt.show()
    print(rvMF(5, mu, s[-1]))

