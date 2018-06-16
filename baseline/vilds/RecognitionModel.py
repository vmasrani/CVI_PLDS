"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import theano
import lasagne
from theano import tensor as T, function, printing
import theano.tensor.nlinalg as Tla
import numpy as np
import theano.tensor.slinalg as Tsla
from sym_blk_tridiag_inv import *
from blk_tridiag_chol_tools import *

class RecognitionModel(object):
    '''
    Recognition Model Interace Class

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor must take the Input Theano variable and create the
    appropriate sampling expression.
    '''

    def __init__(self,Input,xDim,yDim,srng = None,nrng = None):
        self.srng = srng
        self.nrng = nrng

        self.xDim = xDim
        self.yDim = yDim
        self.Input = Input

    def evalEntropy(self):
        '''
        Evaluates entropy of posterior approximation

        H(q(x))

        This is NOT normalized by the number of samples
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def getParams(self):
        '''
        Returns a list of Theano objects that are parameters of the
        recognition model. These will be updated during learning
        '''
        return self.params

    def getSample(self):
        '''
        Returns a Theano object that are samples from the recognition model
        given the input
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def setTrainingMode(self):
        '''
        changes the internal state so that `getSample` will possibly return
        noisy samples for better generalization
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def setTestMode(self):
        '''
        changes the internal state so that `getSample` will supress noise
        (e.g., dropout) for prediction
        '''
        raise Exception("Please implement me. This is an abstract method.")


class SmoothingLDSTimeSeries(RecognitionModel):
    '''
    A "smoothing" recognition model. The constructor accepts neural networks which are used to parameterize mu and Sigma.

    x ~ N( mu(y), sigma(y) )

    '''

    def __init__(self,RecognitionParams,Input,xDim,yDim,srng = None,nrng = None):
        '''
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        '''
        super(SmoothingLDSTimeSeries, self).__init__(Input,xDim,yDim,srng,nrng)

        self.Tt = Input.shape[0]

        # This is the neural network that parameterizes the state mean, mu
        self.NN_Mu = RecognitionParams['NN_Mu']['network']
        # Mu will automatically be of size [T x xDim]
        self.Mu = lasagne.layers.get_output(self.NN_Mu, inputs = self.Input)

        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']
        lambda_net_out = lasagne.layers.get_output(self.NN_Lambda, inputs=self.Input)
        # Lambda will automatically be of size [T x xDim x xDim]
        self.LambdaChol = T.reshape(lambda_net_out, [self.Tt, xDim, xDim]) #+ T.eye(self.xDim)

        self._initialize_posterior_distribution(RecognitionParams)

    def _initialize_posterior_distribution(self, RecognitionParams):

        # Now actually compute the precisions (from their square roots)
        self.Lambda = T.batched_dot(self.LambdaChol, self.LambdaChol.dimshuffle(0,2,1))

        # dynamics matrix & initialize the innovations precision, xDim x xDim
        self.A         = theano.shared(value=RecognitionParams['A'].astype(theano.config.floatX)        ,name='A'        )
        self.QinvChol  = theano.shared(value=RecognitionParams['QinvChol'].astype(theano.config.floatX) ,name='QinvChol' )
        self.Q0invChol = theano.shared(value=RecognitionParams['Q0invChol'].astype(theano.config.floatX),name='Q0invChol')

        self.Qinv  = T.dot(self.QinvChol,self.QinvChol.T)
        self.Q0inv = T.dot(self.Q0invChol,self.Q0invChol.T)

        ################## put together the total precision matrix ######################

        AQinvA = T.dot(T.dot(self.A.T, self.Qinv), self.A)

        # for now we (suboptimally) replicate a bunch of times
        AQinvrep = Tsla.kron(T.ones([self.Tt-1,1,1]),-T.dot(self.A.T, self.Qinv)) # off-diagonal blocks (upper triangle)

        AQinvArep = Tsla.kron(T.ones([self.Tt-2,1,1]), AQinvA+self.Qinv)
        AQinvArepPlusQ = T.concatenate([T.shape_padleft(self.Q0inv + AQinvA), AQinvArep, T.shape_padleft(self.Qinv)])

        # This is our inverse covariance matrix: diagonal (AA) and off-diagonal (BB) blocks.
        self.AA = self.Lambda + AQinvArepPlusQ
        self.BB = AQinvrep

        # symbolic recipe for computing the the diagonal (V) and
        # off-diagonal (VV) blocks of the posterior covariance
        self.V, self.VV, self.S = compute_sym_blk_tridiag(self.AA, self.BB)

        # now compute the posterior mean
        LambdaMu = T.batched_dot(self.Lambda, self.Mu) # scale by precision (no need for transpose; lambda is symmetric)

        #self.old_postX = compute_sym_blk_tridiag_inv_b(self.S,self.V,LambdaMu) # apply inverse

        # compute cholesky decomposition
        self.the_chol = blk_tridag_chol(self.AA, self.BB)
        # intermediary (mult by R^T) -
        ib = blk_chol_inv(self.the_chol[0], self.the_chol[1], LambdaMu)
        # final result (mult by R)-
        self.postX = blk_chol_inv(self.the_chol[0], self.the_chol[1], ib, lower=False, transpose=True)

        # The determinant of the covariance is the square of the determinant of the cholesky factor.
        # Determinant of the Cholesky factor is the product of the diagonal elements of the block-diagonal.
        def comp_log_det(L):
            return T.log(T.diag(L)).sum()
        self.ln_determinant = -2*theano.scan(fn=comp_log_det, sequences=self.the_chol[0])[0].sum()

    def getSample(self):
        normSamps = self.srng.normal([self.Tt, self.xDim])
        return self.postX + blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)

    def evalEntropy(self): # we want it to be smooth, this is a prior on being smooth...
        return self.ln_determinant/2 + self.xDim*self.Tt/2.0*(1+np.log(2*np.pi))

    def getDynParams(self):
        return [self.A]+[self.QinvChol]+[self.Q0invChol]

    def getParams(self):
        return self.getDynParams() + lasagne.layers.get_all_params(self.NN_Mu) + lasagne.layers.get_all_params(self.NN_Lambda)


    def get_summary(self, yy):
        out = {}
        out['xsm'] = np.asarray(self.postX.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['Vsm'] = np.asarray(self.V.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['VVsm'] = np.asarray(self.VV.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['Mu'] = np.asarray(self.Mu.eval({self.Input:yy}), dtype=theano.config.floatX)
        return out

class SmoothingTimeSeries(RecognitionModel):
    '''
    A "smoothing" recognition model. The constructor accepts neural networks which are used to parameterize mu and Sigma.

    x ~ N( mu(y), sigma(y) )

    '''

    def __init__(self,RecognitionParams,Input,xDim,yDim,srng = None,nrng = None):
        '''
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        '''
        super(SmoothingTimeSeries, self).__init__(Input,xDim,yDim,srng,nrng)

#        print RecognitionParams

        self.Tt = Input.shape[0]
        # These variables allow us to control whether the network is deterministic or not (if we use Dropout)
        self.mu_train = RecognitionParams['NN_Mu']['is_train']
        self.lambda_train = RecognitionParams['NN_Lambda']['is_train']

        # This is the neural network that parameterizes the state mean, mu
        self.NN_Mu = RecognitionParams['NN_Mu']['network']
        # Mu will automatically be of size [T x xDim]
        self.Mu = lasagne.layers.get_output(self.NN_Mu, inputs = self.Input)

        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']
        lambda_net_out = lasagne.layers.get_output(self.NN_Lambda, inputs=self.Input)
        self.NN_LambdaX = RecognitionParams['NN_LambdaX']['network']
        lambdaX_net_out = lasagne.layers.get_output(self.NN_LambdaX, inputs=T.concatenate([self.Input[:-1], self.Input[1:]], axis=1))
        # Lambda will automatically be of size [T x xDim x xDim]
        self.AAChol = T.reshape(lambda_net_out, [self.Tt, xDim, xDim]) + T.eye(xDim)
        self.BBChol = T.reshape(lambdaX_net_out, [self.Tt-1, xDim, xDim]) #+ 1e-6*T.eye(xDim)

        self._initialize_posterior_distribution(RecognitionParams)

    def _initialize_posterior_distribution(self, RecognitionParams):

        ################## put together the total precision matrix ######################

        # Diagonals must be PSD
        diagsquare = T.batched_dot(self.AAChol, self.AAChol.dimshuffle(0,2,1))
        odsquare = T.batched_dot(self.BBChol, self.BBChol.dimshuffle(0,2,1))
        self.AA = diagsquare + T.concatenate([T.shape_padleft(T.zeros([self.xDim,self.xDim])), odsquare]) + 1e-6*T.eye(self.xDim)
        self.BB = T.batched_dot(self.AAChol[:-1], self.BBChol.dimshuffle(0,2,1))

        # compute Cholesky decomposition
        self.the_chol = blk_tridag_chol(self.AA, self.BB)

        # symbolic recipe for computing the the diagonal (V) and
        # off-diagonal (VV) blocks of the posterior covariance
        self.V, self.VV, self.S = compute_sym_blk_tridiag(self.AA, self.BB)
        self.postX = self.Mu

        # The determinant of the covariance is the square of the determinant of the cholesky factor (twice the log).
        # Determinant of the Cholesky factor is the product of the diagonal elements of the block-diagonal.
        def comp_log_det(L):
            return T.log(T.diag(L)).sum()
        self.ln_determinant = -2*theano.scan(fn=comp_log_det, sequences=self.the_chol[0])[0].sum()

    def getSample(self):
        normSamps = self.srng.normal([self.Tt, self.xDim])
        return self.postX + blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False, transpose=True)

    def evalEntropy(self):
        return self.ln_determinant/2 + self.xDim*self.Tt/2.0*(1+np.log(2*np.pi))

    def getParams(self):
        return lasagne.layers.get_all_params(self.NN_Mu) + lasagne.layers.get_all_params(self.NN_Lambda) + lasagne.layers.get_all_params(self.NN_LambdaX)

    def get_summary(self, yy):
        out = {}
        out['xsm'] = np.asarray(self.postX.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['Vsm'] = np.asarray(self.V.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['VVsm'] = np.asarray(self.VV.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['Mu'] = np.asarray(self.Mu.eval({self.Input:yy}), dtype=theano.config.floatX)

class MeanFieldGaussian(RecognitionModel):
    '''
    Define a mean field variational approximate posterior (Recognition Model). Here,
    "mean field" is over time, so that for x = (x_1, \dots, x_t, \dots, x_T):

    x ~ \prod_{t=1}^T N( mu_t(y_t), sigma_t(y_t) ).

    Each covariance sigma_t is a full [n x n] covariance matrix (where n is the size
    of the latent space).

    '''

    def __init__(self,RecognitionParams,Input,xDim,yDim,srng = None,nrng = None):
        '''
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x), observation (y)
        '''
        super(MeanFieldGaussian, self).__init__(Input,xDim,yDim,srng,nrng)
        self.Tt = Input.shape[0]
        self.mu_train = RecognitionParams['NN_Mu']['is_train']
        self.NN_Mu = RecognitionParams['NN_Mu']['network']
        self.postX = lasagne.layers.get_output(self.NN_Mu, inputs = self.Input)

        self.lambda_train = RecognitionParams['NN_Lambda']['is_train']
        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']

        lambda_net_out = lasagne.layers.get_output(self.NN_Lambda, inputs=self.Input)
        self.LambdaChol = T.reshape(lambda_net_out, [self.Tt, xDim, xDim])

    def getParams(self):
        network_params = lasagne.layers.get_all_params(self.NN_Mu) + lasagne.layers.get_all_params(self.NN_Lambda)
        return network_params

    def evalEntropy(self):
        def compTrace(Rt):
            return T.log(T.abs_(T.nlinalg.det(Rt))) # 1/2 the log determinant
        theDet,updates = theano.scan(fn=compTrace, sequences=[self.LambdaChol])
        return theDet.sum() + self.xDim*self.Tt/2.0*(1+np.log(2*np.pi))

    def getSample(self):

        normSamps = self.srng.normal([self.Tt, self.xDim])

        def SampOneStep(SampRt, nsampt):
            return T.dot(nsampt,SampRt.T)
        retSamp  = theano.scan(fn=SampOneStep,sequences=[self.LambdaChol, normSamps])[0]
        return retSamp+self.postX

    def get_summary(self, yy):
        out = {}
        out['xsm'] = numpy.asarray(self.postX.eval({self.Input:yy}), dtype=theano.config.floatX)
        V = T.batched_dot(self.LambdaChol, self.LambdaChol.dimshuffle(0,2,1))
        out['Vsm'] = numpy.asarray(V.eval({self.Input:yy}), dtype=theano.config.floatX)
        out['VVsm'] = np.zeros([yy.shape[0]-1, self.xDim, self.xDim]).astype(theano.config.floatX)
        return out
