# coding: utf-8
import numpy as np
import os
from learn_kalman import learn_kalman
from kalman_filter import kalman_smoother, kalman_filter
from cvi_helpers import get_elbo, E_log_p_mc, make_y_R_tilde, sample_posterior, maximize_non_conjugate
from utils import dotdict, plot_posterior, plot_learned_matrices, print_status
from get_data import get_poission_model, get_parameters
import tensorflow as tf
# Supress TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(args):

    # ========================= #
    # --------Initialize------- #
    # ========================= #

    # Get dimensions
    ls, os, T = args.ls, args.os, args.T

    # Define inital matrices
    if args.learn_matrices:
        A = np.eye(ls, ls)
        C = np.eye(os, ls)
        Q = np.eye(ls)
        D = np.eye(os, 1)
        initx = np.eye(ls, 1)
        initV = np.eye(ls)
    else:
        A = args.A_true
        C = args.C_true
        Q = args.Q_true
        D = args.D_true
        initx = args.initx_true
        initV = args.initV_true

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

    # print
    print_status(args)

    elbo = []
    all_loglik = []
    for i in range(args.cvi_iters):

        # -----------E Step -----------
        # Step 3 in Alg 1: compute SG of the non-conjugate part
        # Generate MC samples
        Z = sample_posterior(xfilt, Vfilt, args.nSamples)
        fb, df, dv = E_log_p_mc(args.y_data, Z, C, D)

        # Compute lam1, lam2
        mean_par = np.matmul(C, xfilt) + D
        tlam_1 = (1 - beta) * tlam_1 + (beta) * \
            (df - 2 * (np.multiply(dv, mean_par)))
        tlam_2 = (1 - beta) * tlam_2 + (beta) * (dv)

        # Form y_tilde, R_tilde
        y_tilde, R_tilde = make_y_R_tilde(tlam_1, tlam_2)

        # Update xfilt, Vfilt
        xfilt, Vfilt, VVfilt, loglik = kalman_filter(
            y_tilde, A, C, Q, R_tilde, initx, initV)

        # Get smoothed posteriors (for plotting + M step)
        xsmooth, Vsmooth, VVsmooth, loglik = kalman_smoother(
            y_tilde, A, C, Q, R_tilde, initx, initV)
        # ------------ end -----------

        # -----------M Step -----------
        if args.learn_matrices:
            R = None

            # Run one step of EM algorithm to update other parameters
            A, _, Q, _, initx, initV, LL = learn_kalman(args.y_data, A, C, Q, R, initx, initV,
                                                        max_iter=1,
                                                        smoother=[
                                                            xsmooth, Vsmooth, VVsmooth, loglik],
                                                        fix_R='full',
                                                        fix_C='full',
                                                        verbose=False)

            Z = sample_posterior(xsmooth, Vsmooth, args.nSamples)

            data = {"Z": Z,
                    "Y": args.y_data,
                    "C": C,
                    "D": D}

            C, D = maximize_non_conjugate(
                data, lr=args.mstep_lr, iters=args.mstep_iters, verbose=args.verbose)
        # ------------ end -----------

        # Save elbo + loglik
        elbo += [get_elbo(fb, y_tilde, xfilt, Vfilt, R_tilde, C, D, loglik)]
        all_loglik += [loglik]

        print "Iteration %i, LogLik: %f" % (i, loglik)

    # Plot ELBO
    args.elbo = elbo
    args.loglik = all_loglik
    args.xsmooth = xfilt
    args.Vsmooth = Vsmooth

    if args.learn_matrices:
        args.A_learned = A
        args.C_learned = C
        args.Q_learned = Q
        args.D_learned = D
        args.initx_learned = initx
        args.initV_learned = initV
        plot_posterior(args, path='plots/results_learned_parameters.png')
        plot_learned_matrices(args)
    else:
        plot_posterior(args, path='plots/results_known_parameters.png')



if __name__ == '__main__':
    # Hyperparameters
    args = dotdict()
    args.ls = 5
    args.os = 10
    args.T  = 500
    args.cvi_iters = 15
    args.mstep_lr  = 0.05
    args.mstep_iters = 200
    args.nSamples = 500
    args.verbose  = False
    args.seed     = 10

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
    A, C, Q, D, initx, initV = get_parameters(true_model)
    args.A_true = A
    args.C_true = C
    args.Q_true = Q
    args.D_true = D
    args.initx_true = initx
    args.initV_true = initV

    # Train
    args.learn_matrices = False
    train(args)

    # Learning matrices
    args.learn_matrices = True
    train(args)


