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


import theano
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg as Tla
import theano.tensor.slinalg as Tsla

def blk_tridag_chol(A, B):
    '''
    Compute the cholesky decompoisition of a symmetric, positive definite
    block-tridiagonal matrix.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block 
        off-diagonal matrix

    Outputs: 
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky

    '''

    # Code for computing the cholesky decomposition of a symmetric block tridiagonal matrix
    def compute_chol(Aip1, Bi, Li, Ci):
        Ci = T.dot(Bi.T, Tla.matrix_inverse(Li).T)
        Dii = Aip1 - T.dot(Ci, Ci.T)
        Lii = Tsla.cholesky(Dii)
        return [Lii,Ci]

    L1 = Tsla.cholesky(A[0])
    C1 = T.zeros_like(B[0])

    # this scan returns the diagonal and off-diagonal blocks of the cholesky decomposition
    mat, updates = theano.scan(fn=compute_chol, sequences=[A[1:], B], outputs_info=[L1,C1])

    mat[0] = T.concatenate([T.shape_padleft(L1), mat[0]])
    return mat


def blk_chol_inv(A, B, b, lower = True, transpose = False):
    '''
    Solve the equation Cx = b for x, where C is assumed to be a 
    block-bi-diagonal matrix ( where only the first (lower or upper) 
    off-diagonal block is nonzero.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower) 
        1st block off-diagonal matrix
    
    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the 
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve 
          the problem C^T x = b with a representation of C.) 

    Outputs: 
    x - solution of Cx = b

    '''    
    if transpose:
        A = A.dimshuffle(0, 2, 1)
        B = B.dimshuffle(0, 2, 1)
    if lower:
        x0 = Tla.matrix_inverse(A[0]).dot(b[0])
        def lower_step(Akp1, Bk, bkp1, xk):
            return Tla.matrix_inverse(Akp1).dot(bkp1-Bk.dot(xk))
        X = theano.scan(fn = lower_step, sequences=[A[1:], B, b[1:]], outputs_info=[x0])[0]
        X = T.concatenate([T.shape_padleft(x0), X])
    else:
        xN = Tla.matrix_inverse(A[-1]).dot(b[-1])
        def upper_step(Akm1, Bkm1, bkm1, xk):
            return Tla.matrix_inverse(Akm1).dot(bkm1-(Bkm1).dot(xk))
        X = theano.scan(fn = upper_step, sequences=[A[:-1][::-1], B[::-1], b[:-1][::-1]], outputs_info=[xN])[0]
        X = T.concatenate([T.shape_padleft(xN), X])[::-1]
    return X


def blk_chol_mtimes(A, B, x, lower = True, transpose = False):
    '''
    Evaluate Cx = b, where C is assumed to be a 
    block-bi-diagonal matrix ( where only the first (lower or upper) 
    off-diagonal block is nonzero.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower) 
        1st block off-diagonal matrix
    
    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the 
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve 
          the problem C^T x = b with a representation of C.) 

    Outputs: 
    b - result of Cx = b

    '''    
    if transpose:
        A = A.dimshuffle(0, 2, 1)
        B = B.dimshuffle(0, 2, 1)
    if lower:
        b0 = (A[0]).dot(x[0])
        def lower_step(Ak, Bkm1, xkm1, xk):
            return  Bkm1.dot(xkm1) + Ak.dot(xk)       
        X = theano.scan(fn = lower_step, sequences=[A[1:], B, dict(input=x, taps=[-1, 0])])[0]
        X = T.concatenate([T.shape_padleft(b0), X])    
    else:    
        def lower_step(Ak, Bk, xkm1, xk):
            return  Ak.dot(xkm1) + Bk.dot(xk)       
        X = theano.scan(fn = lower_step, sequences=[A, B, dict(input=x, taps=[-1, 0])])[0]
        bN = (A[-1]).dot(x[-1]) 
        X = T.concatenate([X, T.shape_padleft(bN)])    
    return X


if __name__ == "__main__":
    print 'oh yeah....'


    # Build a block tridiagonal matrix

    npA = np.mat('1  .9; .9 4', dtype=theano.config.floatX)
    npB = .01*np.mat('2  7; 7 4', dtype=theano.config.floatX)
    npC = np.mat('3  0; 0 1', dtype=theano.config.floatX)
    npD = .01*np.mat('7  2; 9 3', dtype=theano.config.floatX)
    npE = .01*np.mat('2  0; 4 3', dtype=theano.config.floatX)
    npF = .01*np.mat('1  0; 2 7', dtype=theano.config.floatX)
    npG = .01*np.mat('3  0; 8 1', dtype=theano.config.floatX)

    npZ = np.mat('0 0; 0 0')

    lowermat = np.bmat([[npF,     npZ, npZ,   npZ],
                           [npB.T,   npC, npZ,   npZ],
                           [npZ,   npD.T, npE,   npZ],
                           [npZ,     npZ, npB.T, npG]])
    print lowermat

    # make lists of theano tensors for the diagonal and off-diagonal blocks
    tA = theano.shared(value=npA)
    tB = theano.shared(value=npB)
    tC = theano.shared(value=npC)
    tD = theano.shared(value=npD)
    tE = theano.shared(value=npE)
    tF = theano.shared(value=npF)
    tG = theano.shared(value=npG)

    theD = T.stack(tF, tC, tE, tG)
    theOD = T.stack(tB.T, tD.T, tB.T)

    npb = np.mat('1 2; 3 4; 5 6; 7 8')
    print npb
    tb = T.matrix('b')

    cholmat = lowermat.dot(lowermat.T)

    # invert matrix using Cholesky decomposition
    # intermediary -
    ib = blk_chol_inv(theD, theOD, tb)
    # final result -
    x = blk_chol_inv(theD, theOD, ib, lower=False, transpose=True)

    f = theano.function([tb], x)

    print 'Cholesky inverse matches numpy inverse: ', np.allclose(f(npb).flatten(), np.linalg.inv(cholmat).dot(np.array([1, 2, 3, 4, 5, 6, 7, 8])))
