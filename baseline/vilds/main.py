import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import lasagne
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
import seaborn as sns

import cPickle
import sys

theano.config.optimizer = 'fast_compile'
msrng = RandomStreams(seed=1)
mnrng = np.random.RandomState(seed=1)

# Load our code
# Add all the paths that should matter right now

from GenerativeModel import *       # Class file for generative models.
from RecognitionModel import *      # Class file for recognition models
# The meat of the algorithm - define the ELBO and initialize Gen/Rec model
from SGVB import *
from sklearn.utils import check_random_state


########################################
# Define a helper class to help us iterate through the training data
class DatasetMiniBatchIndexIterator(object):
    """ Basic mini-batch iterator """

    def __init__(self, y, batch_size=100, randomize=False):
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.rng = np.random.RandomState(np.random.randint(12039210))

    def __iter__(self):
        n_samples = self.y.shape[0]
        # if n_samples == self.batch_size:
        #    yield [self.y, np.arange(n_samples)]
        if self.randomize:
            for _ in xrange(n_samples / self.batch_size):
                if self.batch_size > 1:
                    i = int(self.rng.rand(1) *
                            ((n_samples - self.batch_size - 1)))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                ii = np.arange(i, i + self.batch_size)
                yield [self.y[ii], ii]
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                ii = np.arange(i * self.batch_size, (i + 1) * self.batch_size)
                yield [self.y[ii], ii]


def train(args):
    # From: https://github.com/earcher/vilds/blob/master/code/tutorial.ipynb
    # Let's define a PLDS GenerativeModel. First, choose dimensionality of latent space and output:
    print("=========================")
    print("-----Training VILDS------")
    print("=========================")
    xDim = args.ls
    yDim = args.os

    x_data, y_data = args.x_data.T, args.y_data.T

    ########################################
    # Describe network for mapping into means

    NN_Mu = lasagne.layers.InputLayer((None, yDim))
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, 25, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    #--------------------------------------
    # let's initialize the first layer to have 0 mean wrt our training data
    W0 = np.asarray(NN_Mu.W.get_value(), dtype=theano.config.floatX)
    NN_Mu.W.set_value((W0 / np.dot(y_data, W0).std(axis=0)
                      ).astype(theano.config.floatX))
    W0 = np.asarray(NN_Mu.W.get_value(), dtype=theano.config.floatX)
    b0 = (-np.dot(y_data, W0).mean(axis=0)).astype(theano.config.floatX)
    NN_Mu.b.set_value(b0)
    #--------------------------------------
    NN_Mu = lasagne.layers.DenseLayer(
        NN_Mu, xDim, nonlinearity=linear, W=lasagne.init.Normal())
    NN_Mu.W.set_value(NN_Mu.W.get_value() * 10)
    NN_Mu = dict([('network', NN_Mu)])

    ########################################
    # Describe network for mapping into Covariances
    NN_Lambda = lasagne.layers.InputLayer((None, yDim))
    NN_Lambda = lasagne.layers.DenseLayer(
        NN_Lambda, 25, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    #--------------------------------------
    # let's initialize the first layer to have 0 mean wrt our training data
    W0 = np.asarray(NN_Lambda.W.get_value(), dtype=theano.config.floatX)
    NN_Lambda.W.set_value(
        (W0 / np.dot(y_data, W0).std(axis=0)).astype(theano.config.floatX))
    W0 = np.asarray(NN_Lambda.W.get_value(), dtype=theano.config.floatX)
    b0 = (-np.dot(y_data, W0).mean(axis=0)).astype(theano.config.floatX)
    NN_Lambda.b.set_value(b0)
    #--------------------------------------
    NN_Lambda = lasagne.layers.DenseLayer(
        NN_Lambda, xDim * xDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_Lambda.W.set_value(NN_Lambda.W.get_value() * 10)
    NN_Lambda = dict([('network', NN_Lambda)])

    ########################################
    # define dictionary of recognition model parameters
    recdict = dict([('A', .9 * np.eye(xDim)),
                    # np.linalg.cholesky(np.linalg.inv(np.array(tQ)))),
                    ('QinvChol',  np.eye(xDim)),
                    # np.linalg.cholesky(np.linalg.inv(np.array(tQ0)))),
                    ('Q0invChol', np.eye(xDim)),
                    ('NN_Mu', NN_Mu),
                    ('NN_Lambda', NN_Lambda),
                    ])

    ########################################
    # We can instantiate a recognition model alone and sample from it.
    # First, we have to define a Theano dummy variable for the input observations the posterior expects:
    Y = T.matrix()

    rec_model = SmoothingLDSTimeSeries(
        recdict, Y, xDim, yDim, srng=msrng, nrng=mnrng)
    rsamp = rec_model.getSample()

        # initialize training with a random generative model (that we haven't generated data from):
    initGenDict = dict([
                ('output_nlin', 'softplus')
                    ])

    # Instantiate an SGVB class:
    sgvb = SGVB(initGenDict, PLDS, recdict,
                SmoothingLDSTimeSeries, xDim=xDim, yDim=yDim)

    ########################################
    # Define a bare-bones thenao training function
    batch_y = T.matrix('batch_y')

    ########################################
    # choose learning rate and batch size
    learning_rate = 1e-2
    batch_size = 100

    ########################################
    # use lasagne to get adam updates
    updates = lasagne.updates.adam(-sgvb.cost(),
                                   sgvb.getParams(), learning_rate=learning_rate)

    ########################################
    # Finally, compile the function that will actually take gradient steps.
    train_fn = theano.function(
            outputs=sgvb.cost(),
            inputs=[theano.In(batch_y)],
            updates=updates,
            givens={sgvb.Y: batch_y},
        )

    ########################################
    # set up an iterator over our training data
    yiter = DatasetMiniBatchIndexIterator(
        y_data, batch_size=batch_size, randomize=True)

    ########################################
    # Iterate over the training data for the specified number of epochs
    cost = []
    for ie in np.arange(args.iters):
        print('--> entering epoch %d' % ie)
        for y, _ in yiter:
            cost.append(train_fn(y))

    #########################
    # Since the model is non-identifiable, let's find the best linear projection from the
    # learned posterior mean into the 'true' training-data latents
    pM = sgvb.mrec.postX.eval({sgvb.Y: y_data})
    wgt = np.linalg.lstsq(pM-pM.mean(), x_data-x_data.mean())[0]

    #########################
    # sample from the trained recognition model
    rtrain_samp = sgvb.mrec.getSample()

    # #########################
    # # plot 25 samples from the posterior
    # for idx in np.arange(25): # plot multiple samples from the posterior
    #     xs = rtrain_samp.eval({sgvb.Y: y_test})
    #     plt.plot(np.dot(xs,wgt),'k')

    # and now plot the posterior mean
    pMtest = sgvb.mrec.postX.eval({sgvb.Y: y_data})
    x_hat = np.dot(pMtest,wgt)

    return x_hat.T
    # plt.close('all')
    # fig = plt.figure(figsize=(16, 12))
    # sns.set()
    # plt.style.use('seaborn')

    # for i in range(4):
    #     ax = fig.add_subplot(3, 2, 3 + i)
    #     plt_true = plt.plot(x_data[:, i], color='C1', label='True')
    #     plt_post = plt.plot(x_hat[:, i], color='C2', label='VILDS')
    #     # plt.legend(handles = plt_post + plt_true)
    #     plt.xlabel('time')
    #     plt.title('samples from the trained approximate posterior')
    # plt.savefig('plots/vilds.png')

if __name__ == "__main__":
    run()



