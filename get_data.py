import theano
import lasagne
import numpy as np
from matplotlib import pyplot as plt
from lasagne.nonlinearities import linear
from theano.tensor.shared_randomstreams import RandomStreams
from poission_model import PLDS

def get_poission_model(xDim, yDim, seed=1):
    """ This code is modified from the notebook here: https://github.com/earcher/vilds"""

    msrng = RandomStreams(seed=seed)
    mnrng = np.random.RandomState(seed=seed)

    # Define a neural network that maps the latent state into the output
    gen_nn = lasagne.layers.InputLayer((None, xDim))
    gen_nn = lasagne.layers.DenseLayer(
        gen_nn, yDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_XtoY_Params = dict([('network', gen_nn)])

    #Put all the parameters in a dictionary
    gendict = dict([('A', 0.8 * np.eye(xDim)),         # Linear dynamics parameters
                    ('QChol', 2 * np.diag(np.ones(xDim))),  # innovation noise
                    ('Q0Chol', 2 * np.diag(np.ones(xDim))),
                    ('x0', np.zeros(xDim)),
                    #                ('RChol', np.ones(yDim)),             # observation covariance
                    # neural network output mapping
                    ('NN_XtoY_Params', NN_XtoY_Params),
                    # ('output_nlin', 'exponential')  # for poisson observations
                    ('output_nlin', 'softplus')  # for poisson observations
                    ])
    # import ipdb; ipdb.set_trace()
    # Instantiate a PLDS generative model:
    true_model = PLDS(gendict, xDim, yDim, srng=msrng, nrng=mnrng)

    return true_model

def get_parameters(model):
    A, QChol, Q0Chol, x0, W, b = model.getParams()
    A = A.get_value()
    C = W.get_value().T
    Q = np.matmul(QChol.get_value(), QChol.get_value().T)
    D = np.expand_dims(b.get_value(), 1)
    initx = np.expand_dims(x0.get_value(), 1)
    initV = np.matmul(Q0Chol.get_value(), Q0Chol.get_value().T)

    return A, C, Q, D, initx, initV

