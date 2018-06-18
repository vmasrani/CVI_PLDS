import numpy as np
import tensorflow as tf
from data.get_data import get_data, get_parameters
from cvi.cvi import train as train_cvi
from cvi.utils import dotdict, plot_cvi, plot_baselines, plot_elbo_loglik, print_mse
from baseline.gauss import train as train_gauss
from baseline.vilds.main import train as train_vilds
import os
# Supress TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run():
    # Hyperparameters
    args = dotdict()
    args.ls        = 5
    args.os        = 10
    args.T         = 300
    args.iters     = 15
    args.nSamples  = 500
    args.verbose   = False
    args.seed      = 0

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Get data
    x_data, y_data, model = get_data(args.ls, args.os, args.T)
    args.x_data = x_data
    args.y_data = y_data

    # Get true parameters
    A, C, Q, initx, initV = get_parameters(model)
    args.A = A
    args.C = C
    args.Q = Q
    args.initx = initx
    args.initV = initV

    # Train all
    results = train_cvi(args)
    results["true"]  = x_data
    results["vilds"] = train_vilds(args)
    results["gauss"] = train_gauss(args)

    # Evaluate
    plot_elbo_loglik(args, results)
    plot_cvi(args, results)
    plot_baselines(results)
    print_mse(results)

if __name__ == '__main__':
    run()
