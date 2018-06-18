import matplotlib.pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d
import seaborn as sns

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Sample from MVN using cholesky
def chol_sample(noise, mu, cov):
    # Shape of noise parameter determines how many samples are returned
    assert mu.shape[0] == cov.shape[0] == cov.shape[1]
    if mu.ndim < 2:
        mu = np.expand_dims(mu, -1)
    L = np.linalg.cholesky(cov)
    return (np.matmul(L, noise) + mu)


# Make column vector
def column_vec(v):
    return v.reshape(-1, 1)


# Trace without computing off-diagonal entries in the matrix product
def trace(A, B):
    return np.sum(inner1d(A, B.T))

def plot_elbo_loglik(args, results, path='plots/elbo_loglik.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(16, 4))
    sns.set()
    plt.style.use('seaborn')

    T = args.T
    elbo = [-i for i in results["elbo"]]
    loglik = [-i for i in results["loglik"]]

    # Elbo
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(elbo)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('ELBO')
    ax.set_title('Elbo Convergence')

    # loglik
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(loglik)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('LogLik')
    ax.set_title('LogLik Convergence')

    # Save
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path)

def plot_cvi(args, results, path='plots/cvi_results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(16, 12))
    sns.set()
    plt.style.use('seaborn')

    T = args.T
    x_data = results["true"]
    xsmooth = results["xsmooth"]
    uncertainty = results["Vsmooth"]

    # labels
    label = [r'$x_{t,1}$', r'$x_{t,2}$', r'$x_{t,3}$', r'$x_{t,4}$', ]

    for i in range(0, min(args.ls, 4)):
        # latent_dim 1
        ax = fig.add_subplot(2, 2, 1 + i)
        ax.plot(range(T), x_data[i, :], color='C1', label='True')
        ax.plot(range(T), xsmooth[i, :], color='C2', label='CVI')
        ax.fill_between(range(T), xsmooth[i, :] - uncertainty[i, i, :],
                        xsmooth[i, :] + uncertainty[i, i, :], color='C2', alpha=0.35)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(label[i])
        ax.legend(["True", "CVI"], loc='upper right')

    # plt.suptitle(
    #     'Posterior mean inference compared with ground truth in LDS w/ Poission Likelihoods model')
    # Save
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path)


def plot_baselines(results, path='plots/baselines.png'):
    plt.close('all')
    fig = plt.figure(figsize=(16, 12))
    sns.set()
    plt.style.use('seaborn')
    label = [r'$x_{t,1}$', r'$x_{t,2}$', r'$x_{t,3}$', r'$x_{t,4}$', ]
    for i in range(4):
        ax = fig.add_subplot(2, 2, 1 + i)
        plt_true  = plt.plot(results["true"][i, :],  color='C1', alpha=1.0, label='True')
        plt_cvi   = plt.plot(results["cvi"][i, :],   color='C2', alpha=0.75, label='CVI')
        plt_vilds = plt.plot(results["vilds"][i, :], color='C3', alpha=0.75, label='VILDS')
        plt_gauss = plt.plot(results["gauss"][i, :], color='C4', alpha=0.75, label='EM-Gauss')
        plt.legend(handles = plt_true + plt_cvi + plt_vilds + plt_gauss)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(label[i])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)

def print_mse(results):
    d_cvi   = results['true'] - results['cvi']
    d_vilds = results['true'] - results['vilds']
    d_gauss = results['true'] - results['gauss']

    mse_cvi = np.sqrt(np.sum(d_cvi**2))
    mse_vilds = np.sqrt(np.sum(d_vilds**2))
    mse_gauss = np.sqrt(np.sum(d_gauss**2))


    print("=========================")
    print("----------MSE------------")
    print("=========================")
    print("cvi:  {}".format(mse_cvi))
    print("vilds  {}".format(mse_vilds))
    print("gauss:  {}".format(mse_gauss))
