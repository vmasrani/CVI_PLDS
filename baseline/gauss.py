from kalman.kalman_filter import kalman_smoother
from kalman.learn_kalman import learn_kalman
import numpy as np

def train(args):
    print("=========================")
    print("----- Gaussian EM -------")
    print("=========================")
    # Get dimensions
    ls, os, T = args.ls, args.os, args.T

    # Get inital matrices
    A = args.A
    C = args.C
    Q = args.Q
    initx = args.initx
    initV = args.initV

    R = np.eye(args.os)

    # Learn R
    fix = {
        "fix_A":"full",
        "fix_Q":"full",
        "fix_C":"full",
        "fix_initx":"full",
        "fix_initV":"full"
    }
    _, _, _, R, _, _, _ = learn_kalman(args.y_data, A, C, Q, R, initx, initV, max_iter=50, smoother=kalman_smoother, **fix)

    # Smooth
    xsmooth, _, _, _ = kalman_smoother(args.y_data, A, C, Q, R, initx, initV)

    return xsmooth

