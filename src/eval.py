import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import wishart, multivariate_normal
import math
import imageio
from sklearn.cluster import KMeans
import gibbs_sampler as gs
import DP_gibbs_sampler as dgs
import multivariate_gibbs_sampler as vgs
import DP_multivariate_gibbs_sampler as dvgs
import vMF_gibbs_sampler as vmgs
import DP_vMF_gibbs_sampler as dvmgs
#from sklearn.cluster import Kmeans





def save(z,aari, aami, al, n1, n2, n3, n4):
    aari =  aari/ 10
    aami = aami/10
    al = al/10
    plt.plot(np.arange(len(al)),aari)
    plt.xlabel("iterations")
    plt.ylabel("ARI")
    plt.savefig(n2 + ".png")
    plt.clf()
#    
#    plt.xlabel("iterations")
#    plt.ylabel("AMI")
#    plt.plot(np.arange(len(al)),aami)
#    plt.savefig(n2 + ".png")
#    plt.clf()
#
#    plt.xlabel("iterations")
#    plt.ylabel("log likelihood")
#    plt.plot(np.arange(len(al)),al)
#    plt.savefig(n3 + ".png")
#    plt.clf()
#    
#    u, c = np.unique(z, return_counts=True)
#    disp_order = u[np.argsort(c)[::-1]]
#    plt.bar(disp_order, np.sort(c)[::-1])
#    plt.xlabel("Size")
#    plt.ylabel("clusters")
#    plt.savefig(n4 + ".png")
#    plt.clf()
#
    


def evaluate():
    

    al = np.zeros((1000,))
    aari = np.zeros((1000,))
    aami = np.zeros((1000,))
    for r in range(10):
        x, y = gs.synth_data([0.25, 0.25, 0.25, 0.25], 500)
        z, l, ARI, AMI = gs.gibbs_sampler(x, gt=y, niter=1000)
        al += l
        aari += ARI
        aami += AMI

    save(z,aari, aami, al, "ari_uni", "ami_uni", "log_uni", "cc_uni")
    al = np.zeros((1000,))
    aari = np.zeros((1000,))
    aami = np.zeros((1000,))
    for r in range(10):
        x, y = dgs.synth_data([0.25, 0.25, 0.25, 0.25], 500)
        z, l, ARI, AMI = dgs.DP_gibbs_sampler(x, gt=y, niter=1000)
        al += l
        aari += ARI
        aami += AMI
        

    save(z,aari, aami, al, "ari_DPuni", "ami_DPuni", "log_DPuni", "cc_DPuni")
    al = np.zeros((1000,))
    aari = np.zeros((1000,))
    aami = np.zeros((1000,))
    for r in range(10):
        x, y = vgs.synth_data([0.25, 0.25, 0.25, 0.25], 500, 3)
        z, l, ARI, AMI = vgs.mv_gibbs_sampler(x, gt=y, niter=1000)
        al += l
        aari += ARI
        aami += AMI


    save(z,aari, aami, al, "ari_mv", "ami_mv", "log_mv", "cc_mv")
    al = np.zeros((1000,))
    aari = np.zeros((1000,))
    aami = np.zeros((1000,))
    for r in range(10):
        x, y = dvgs.synth_data([0.25, 0.25, 0.25, 0.25], 500, 3)
        z, l, ARI, AMI = dvgs.DP_mv_gibbs_sampler(x, gt=y, niter=1000)
        al += l
        aari += ARI
        aami += AMI


    save(z,aari, aami, al, "ari_DPmv", "ami_DPmv", "log_DPmv", "cc_DPmv")
    al = np.zeros((1000,))
    aari = np.zeros((1000,))
    aami = np.zeros((1000,))
    for r in range(10):
        x, y = vmgs.synth_data([0.25, 0.25, 0.25, 0.25], 500, 3)
        K = np.unique(y)
        z, l, ARI, AMI = vmgs.vmf_gibbs_sampler(x, gt=y, niter=1000)
        al += l
        aari += ARI
        aami += AMI

    save(z,aari, aami, al, "ari_vmf", "ami_vmf", "log_vmf", "cc_vmf")
    al = np.zeros((1000,))
    aari = np.zeros((1000,))
    aami = np.zeros((1000,))
    for r in range(10):
        x, y = dvmgs.synth_data([0.25, 0.25, 0.25, 0.25], 500, 3)
        K = np.unique(y)
        z, l, ARI, AMI = dvmgs.DP_vmf_gibbs_sampler(x, gt=y, niter=1000)
        al += l
        aari += ARI
        aami += AMI

    save(z,aari, aami, al, "ari_DPvmf", "ami_DPvmf", "log_DPvmf", "cc_DPvmf")





evaluate()
