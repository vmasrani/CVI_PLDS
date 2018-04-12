import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.core.umath_tests import inner1d
import seaborn as sns; sns.set()

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


def check_format(y, mu, cov):
    # Assure y_i in [1,-1]
    y = y > 0
    y = 2 * y - 1
    if y.shape[1] != 1:
        y = y.T

    # MultivariateNormalFullCovariance expects batch index to be first
    if cov.shape[1] != cov.shape[2]:
        cov = np.swapaxes(cov, 0, 2)

    if cov.shape[0] != mu.shape[0]:
        mu = np.swapaxes(mu, 0, 1)

    return y, mu, cov


def plot_posterior(args, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(16, 12))
    mpl.style.use('seaborn')

    T = args.T
    elbo = [-i for i in args.elbo]
    loglik = [-i for i in args.loglik]
    x_data = args.x_data
    xsmooth = args.xsmooth
    uncertainty = args.Vsmooth

    # Elbo
    ax = fig.add_subplot(3, 2, 1)
    ax.plot(elbo)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('ELBO')
    ax.set_title('Elbo Convergence')

    # loglik
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(loglik)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('LogLik')
    ax.set_title('LogLik Convergence')

    for i in range(0, min(args.ls, 4)):
        # latent_dim 1
        ax = fig.add_subplot(3, 2, 3 + i)
        ax.plot(range(T), x_data[i, :], color = 'C1', label='True')
        ax.plot(range(T), xsmooth[i, :], color='C2',label='CVI')
        ax.fill_between(range(T), xsmooth[i, :]-uncertainty[i, i, :], xsmooth[i, :]+uncertainty[i, i, :], color='C2', alpha=0.35)
        ax.set_title('Latent state, dimension {}'.format(str(i)))
        ax.set_xlabel('Timestep t')
        ax.legend(["True", "CVI"], loc='upper right')

    plt.suptitle('Posterior mean inference compared with ground truth in LDS w/ Poission Likelihoods model')
    # Save
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path)

