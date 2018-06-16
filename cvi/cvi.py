# coding: utf-8
import os

import numpy as np
import tensorflow as tf

from cvi_helpers import E_log_p_mc, get_elbo, make_y_R_tilde, sample_posterior
from kalman.kalman_filter import kalman_filter, kalman_smoother


def train(args):
    print("=========================")
    print("-------Training CVI------")
    print("=========================")
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

    # print "===========Running CVI==============="
    for i in range(args.iters):
        # Step 3 in Alg 1: compute SG of the non-conjugate part
        # Generate MC samples
        mc_latent = sample_posterior(xfilt, Vfilt, args.nSamples)
        fb, df, dv = E_log_p_mc(args.y_data, mc_latent, C)

        # Compute lam1, lam2l
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

    # Get smoothed
    xsmooth, Vsmooth, _, _ = kalman_smoother(y_tilde, A, C, Q, R_tilde, initx, initV)

    results = {
        "xsmooth":xsmooth,
        "Vsmooth":Vsmooth,
        "elbo":elbo,
        "loglik":all_loglik,
        "cvi":xsmooth # For comparison against baselines
    }

    return results
