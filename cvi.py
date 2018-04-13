# coding: utf-8
import os

import numpy as np
import tensorflow as tf

from cvi_helpers import E_log_p_mc, get_elbo, make_y_R_tilde, sample_posterior
from get_data import get_parameters, get_poission_model
from kalman_filter import kalman_filter
from utils import dotdict,  plot_posterior

# Supress TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(args):

    # ========================= #
    # --------Initialize------- #
    # ========================= #

    # Get dimensions
    ls, os, T = args.ls, args.os, args.T

    # Get inital matrices
    A = args.A
    C = args.C
    Q = args.Q
    initx = args.initx
    initV = args.initV

    # =========================#
    # --------Run CVI + EM-----#
    # =========================#

    # Initialize lam1,lam2
    tlam_1 = 1e-6 * np.ones((os, T))
    tlam_2 = 1e-6 * -0.5 * np.ones((os, T)) / 2
    beta = 0.9

    # Make initial xfilt, Vfilt
    y_tilde, R_tilde = make_y_R_tilde(tlam_1, tlam_2)
    xfilt, Vfilt, VVfilt, loglik = kalman_filter(
        y_tilde, A, C, Q, R_tilde, initx, initV)

    elbo = []
    all_loglik = []

    print "===========Running CVI==============="
    for i in range(args.cvi_iters):
        # Step 3 in Alg 1: compute SG of the non-conjugate part
        # Generate MC samples
        mc_latent = sample_posterior(xfilt, Vfilt, args.nSamples)
        fb, df, dv = E_log_p_mc(args.y_data, mc_latent, C)

        # Compute lam1, lam2
        mean_par = np.matmul(C, xfilt)
        tlam_1 = (1 - beta) * tlam_1 + (beta) * \
            (df - 2 * (np.multiply(dv, mean_par)))
        tlam_2 = (1 - beta) * tlam_2 + (beta) * (dv)

        # Form y_tilde, R_tilde
        y_tilde, R_tilde = make_y_R_tilde(tlam_1, tlam_2)

        # Update xfilt, Vfilt
        xfilt, Vfilt, VVfilt, loglik = kalman_filter(
            y_tilde, A, C, Q, R_tilde, initx, initV)

        # Save elbo + loglik
        elbo += [get_elbo(fb, y_tilde, xfilt, Vfilt, R_tilde, C, loglik)]
        all_loglik += [loglik]

        print "Iteration %i, LogLik: %f" % (i, loglik)


    # Plot ELBO
    args.elbo = elbo
    args.loglik = all_loglik
    args.xsmooth = xfilt
    args.Vsmooth = Vfilt

    plot_posterior(args, path='plots/results.png')


if __name__ == '__main__':
    # Hyperparameters
    args = dotdict()
    args.ls          = 5
    args.os          = 10
    args.T           = 500
    args.cvi_iters   = 10
    args.nSamples    = 500
    args.verbose     = False
    args.seed        = 2

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make model
    true_model = get_poission_model(args.ls, args.os)

    # sample data
    x_data, y_data = true_model.sampleXY(args.T)
    args.x_data = x_data.T
    args.y_data = y_data.T

    # Get true parameters
    A, C, Q, initx, initV = get_parameters(true_model)
    args.A = A
    args.C = C
    args.Q = Q
    args.initx = initx
    args.initV = initV

    # Train
    train(args)
