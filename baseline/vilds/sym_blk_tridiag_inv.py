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
from theano.ifelse import ifelse
import theano.tensor.nlinalg as Tla

def compute_sym_blk_tridiag(AA, BB, iia = None, iib = None):
    '''
    Symbolically compute block tridiagonal terms of the inverse of a *symmetric* block tridiagonal matrix.
    
    All input & output assumed to be stacked theano tensors. Note that the function expects the off-diagonal
    blocks of the upper triangle & returns the lower-triangle (the transpose). Matrix is assumed symmetric so 
    this doesn't really matter, but be careful. 

    Input: 
    AA - (T x n x n) diagonal blocks 
    BB - (T-1 x n x n) off-diagonal blocks (upper triangle)
    iia - (T x 1) block index of AA for the diagonal
    iib - (T-1 x 1) block index of BB for the off-diagonal

    Output: 
    D  - (T x n x n) diagonal blocks of the inverse
    OD - (T-1 x n x n) off-diagonal blocks of the inverse (lower triangle)
    S  - (T-1 x n x n) intermediary matrix computation used in inversion algorithm 
 
    From: 
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"
    Note: Could be generalized to non-symmetric matrices, but it is not currently implemented.

    (c) Evan Archer, 2015
    '''
    BB = -BB

    ## Set up some parameters
    if iia is None:
        nT = T.shape(AA)[0]
    else:
        nT = T.shape(iia)[0]

    d = T.shape(AA)[1]

    # if we don't have special indexing requirements, just use the obvious indices
    if iia is None:
        iia = T.arange(nT)
    if iib is None:
        iib = T.arange(nT-1)
    
    III = T.eye(d, dtype=theano.config.floatX)

    initS = T.zeros([d,d], dtype=theano.config.floatX)

    def compute_S(idx, Sp1, zAA, zBB):
        Sm = ifelse(T.eq(idx, nT-2), 
                    T.dot(zBB[iib[-1]], Tla.matrix_inverse(zAA[iia[-1]])),
                    T.dot(zBB[iib[idx]],Tla.matrix_inverse(zAA[iia[T.min([idx+1,nT-2])]] 
                    - T.dot(Sp1,T.transpose(zBB[iib[T.min([idx+1,nT-2])]]))))
                )
        return Sm

    S, updates_S = theano.scan(compute_S,
                             sequences=[T.arange(nT-2,-1,-1)], 
                             outputs_info=initS, 
                             non_sequences=[AA,BB])

    initD = T.zeros([d,d], dtype=theano.config.floatX)
    initOD = T.zeros([d,d], dtype=theano.config.floatX)

    def compute_D(idx, Dm1, zS, zAA, zBB):
        D = ifelse(T.eq(idx, nT-1),
                   T.dot(Tla.matrix_inverse(zAA[iia[-1]]), 
		       III + T.dot(T.transpose(zBB[iib[idx-1]]),
			   T.dot(Dm1,S[0])))
                   , 
                   ifelse(T.eq(idx, 0), 
                          Tla.matrix_inverse(zAA[iia[0]]
			      - T.dot(zBB[iib[0]], T.transpose(S[-1]))),
                          T.dot(Tla.matrix_inverse(zAA[iia[idx]] 
                                - T.dot(zBB[iib[T.min([idx,nT-2])]],T.transpose(S[T.max([-idx-1,-nT+1])]))),
			        III + T.dot(T.transpose(zBB[iib[T.min([idx-1,nT-2])]]),
				  T.dot(Dm1,S[-idx])))
                      )
               )
        return D

    D, updates_D = theano.scan(compute_D,
                             sequences=[T.arange(0,nT)], 
                             outputs_info=initD, 
                             non_sequences=[S, AA,BB])
    
    def compute_OD(idx, zS, zD, zAA, zBB):
        OD = T.dot(T.transpose(zS[-idx-1]),zD[idx])
        return OD

    OD, updates_OD = theano.scan(compute_OD,
                              sequences=[T.arange(0,nT-1)], 
                              outputs_info=None, 
                              non_sequences=[S, D, AA, BB])

    return [D, OD, S]#, updates_D+updates_OD+updates_S]


def compute_sym_blk_tridiag_inv_b(S,D,b):
    '''
    Symbolically solve Cx = b for x, where C is assumed to be *symmetric* block matrix.

    Input: 
    D  - (T x n x n) diagonal blocks of the inverse
    S  - (T-1 x n x n) intermediary matrix computation returned by  
         the function compute_sym_blk_tridiag

    Output: 
    x - (T x n) solution of Cx = b 

   From: 
    Jain et al, 2006
  "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"

    (c) Evan Archer, 2015
    '''
    nT = T.shape(b)[0]
    d = T.shape(b)[1]
    initp = T.zeros([d], dtype=theano.config.floatX)
    inity = T.zeros([d], dtype=theano.config.floatX)
    initq = T.zeros([d], dtype=theano.config.floatX)

    def compute_p(idx, pp, b, S):
        pm = ifelse(T.eq(idx, nT-1),
                    b[-1],
                    b[idx] + T.dot(S[T.max([-idx-1,-nT+1])],pp)
        )
        return pm
    
    p, updates = theano.scan(compute_p,
                             sequences=[T.arange(nT-1,-1,-1)],
                             outputs_info=initp,
                             non_sequences=[b,S])
    
    def compute_q(idx, qm, b, S, D):
        qp = ifelse(T.eq(idx, 0),
                    T.dot(T.dot(T.transpose(S[-1]),D[0]), b[0]),
                    T.dot(T.transpose(S[-idx-1]), qm + T.dot(D[idx],b[idx]) )
                )
        return qp
    
    q, updates_q = theano.scan(compute_q,
                             sequences=[T.arange(nT-1)],
                             outputs_info=p[0],
                             non_sequences=[b,S,D])
            
    def compute_y(idx, p, q, S, D):
        yi = ifelse(T.eq(idx, 0),
                    T.dot(D[0], p[-1]),
                    ifelse(T.eq(idx, nT-1),
                           T.dot(D[-1],p[0]) + q[-1],
                           T.dot(D[idx], p[-idx-1]) + q[idx-1]
                       )
        )
        return yi
        
    y, updates_y = theano.scan(compute_y,
                             sequences=[T.arange(nT)],
                             outputs_info=None,
                             non_sequences=[p,q,S,D])
    
    #return [y, updates_q+updates+y]
    return y
                                    
if __name__ == "__main__": 
    # Build a block tridiagonal matrix 
    npA = np.mat('1 6; 6 4', dtype=theano.config.floatX)
    npB = np.mat('2 7; 7 4', dtype=theano.config.floatX)
    npC = np.mat('3 9; 9 1', dtype=theano.config.floatX)
    npD = np.mat('7 2; 9 3', dtype=theano.config.floatX)
    npZ = np.mat('0 0; 0 0')

    # a 2x2 block tridiagonal matrix with 4x4 blocks
    fullmat = np.bmat([[npA,     npB, npZ,   npZ], 
                       [npB.T,   npC, npD,   npZ], 
                       [npZ,   npD.T, npC,   npB], 
                       [npZ,     npZ, npB.T, npC]])

    # make lists of theano tensors for the diagonal and off-diagonal blocks
    tA = theano.shared(value=npA)
    tB = theano.shared(value=npB)
    tC = theano.shared(value=npC)
    tD = theano.shared(value=npD)
    
    AAin = T.stack(tA, tC, tC, tC)
    BBin = T.stack(tB, tD, tB)

    D, OD, S = compute_sym_blk_tridiag(AAin,BBin)

    print D.eval()
    print OD.eval()
    print fullmat.I

    # test solving the linear sysstem Ay=b
    # now let's implement the solver (IE, we want to solve for y in Ay=b)

    npb = np.asmatrix(np.arange(4*2, dtype=theano.config.floatX).reshape((4,2)))
    b = theano.shared(value=npb)
    print npb

    y = compute_sym_blk_tridiag_inv_b(S,D,b)
    print y.eval()

    print np.linalg.pinv(fullmat).dot(npb.reshape(8,1))


    ## This example illustrates how to use the function to succinctly index matrices with repeated blocks
    the_blocks = T.stack(tA, tB, tC, tD)
    iiA = T.ivector()
    iiB = T.ivector()
    
    Dii, ODii, Sii = compute_sym_blk_tridiag(the_blocks, the_blocks, iiA, iiB)
    
    Deval = theano.function([iiA, iiB], Dii)
    ODeval = theano.function([iiA, iiB], ODii)
    

    Diie = Deval([0,2,2,2], [1,3,1])
    ODiie = ODeval([0,2,2,2], [1,3,1])
    print Diie[0] - D[0].eval()
    
    print ODiie[0] - OD[0].eval()
