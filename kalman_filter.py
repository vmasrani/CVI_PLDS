import numpy as np
from scipy.stats import multivariate_normal


def kalman_filter(y, A, C, Q, R, init_x, init_V, return_innovations=False):

    # INPUTS:
    # y(:,t)   - the observation at time t
    # A - the system matrix
    # C - the observation matrix
    # Q - the system covariance
    # R - the observation covariance
    # init_x - the initial state (column) vector
    # init_V - the initial state covariance
    #
    # OUTPUTS (where X is the hidden state being estimated)
    # x(:,t) = E[X(:,t) | y(:,1:t)]
    # V(:,:,t) = Cov[X(:,t) | y(:,1:t)]
    # VV(:,:,t) = Cov[X(:,t), X(:,t-1) | y(:,1:t)] t >= 2
    # loglik = sum{t=1}^T log P(y(:,t))

    os, T = y.shape
    ss = A.shape[0]

    x  = np.zeros([ss, T])
    V  = np.zeros([ss, ss, T])
    VV = np.zeros([ss, ss, T])

    # Duplicate slices of 2d matrices, leave 3d matrix alone
    A, C, Q, R = tensorify(T, A, C, Q, R)

    loglik = 0
    innovations = []
    for t in range(T):
        if t == 0:
            prevx = init_x
            prevV = init_V
            initial = 1
        else:
            prevx = x[:, [t - 1]]
            prevV = V[:, :, t - 1]
            initial = 0

        x[:, [t]], V[:, :, t], VV[:, :, t], LL, inn = kalman_update(A[:, :, t], C[:, :, t], Q[:, :, t], R[
            :, :, t], y[:, [t]], prevx, prevV, initial)
        loglik += LL  # loglik is a scalar,
        innovations += inn  # inn is a list, used to calculate generalization error

    if return_innovations:
        return x, V, VV, loglik, innovations
    else:
        return x, V, VV, loglik


def kalman_predict(A, Q, x, V):
    # INPUTS:
    # A - the system matrix
    # Q - the system covariance
    # x - E[X(t-1) | y(:, 1:t-1)] prior mean
    # V - Cov[X(t-1) | y(:, 1:t-1)] prior covariance
    #
    # OUTPUTS (where X is the hidden state being estimated)
    # xpred =   E[ X | y(:, 1:t-1) ]
    # Vpred = Var[ X(t) | y(:, 1:t-1) ]
    xpred = np.matmul(A, x)
    Vpred = np.matmul(np.matmul(A, V), A.T) + Q
    return xpred, Vpred


def kalman_update(A, C, Q, R, y, x, V, initial):

    # INPUTS:
    # A - the system matrix
    # C - the observation matrix
    # Q - the system covariance
    # R - the observation covariance
    # y(:)   - the observation at time t
    # x(:) - E[X | y(:, 1:t-1)] prior mean
    # V(:,:) - Cov[X | y(:, 1:t-1)] prior covariance
    #
    # OUTPUTS (where X is the hidden state being estimated)
    # xnew(:) =   E[ X | y(:, 1:t) ]
    # Vnew(:,:) = Var[ X(t) | y(:, 1:t) ]
    # VVnew(:,:) = Cov[ X(t), X(t-1) | y(:, 1:t) ]
    # loglik = log P(y(:,t) | y(:,1:t-1)) log-likelihood of innovation

    if initial:
        xpred = x
        Vpred = V
    else:
        # Prediction
        xpred, Vpred = kalman_predict(A, Q, x, V)

    # Get dimensions
    os = y.shape[0]
    ss = V.shape[0]
    # Gain and innovation
    e = y - np.matmul(C, xpred)                            # innovation
    S = np.matmul(np.matmul(C, Vpred), C.T) + R            # innovation covariance
    K = np.matmul(np.matmul(Vpred, C.T), np.linalg.inv(S))  # Kalman gain matrix

    # Correction
    xnew = xpred + np.matmul(K, e)
    Vnew = np.matmul((np.identity(ss) - np.matmul(K, C)), Vpred)
    VVnew = np.matmul(np.matmul((np.identity(ss) - np.matmul(K, C)), A), V)

    # Loglik
    loglik = multivariate_normal.logpdf(e.reshape([-1]), mean=np.zeros(os), cov=S)
    innovation = [e]

    return xnew, Vnew, VVnew, loglik, innovation


def kalman_smoother(y, A, C, Q, R, init_x, init_V):
    os, T = y.shape
    ss = A.shape[0]

    xsmooth  = np.zeros([ss, T])
    Vsmooth  = np.zeros([ss, ss, T])
    VVsmooth = np.zeros([ss, ss, T])

    # Forward pass
    xfilt, Vfilt, VVfilt, loglik = kalman_filter(y, A, C, Q, R, init_x, init_V)

    # Backward pass
    xsmooth[:, [T - 1]]   = xfilt[:, [T - 1]]
    Vsmooth[:, :, T - 1]  = Vfilt[:, :, T - 1]
    VVsmooth[:, :, T - 1] = VVfilt[:, :, T - 1]

    A, Q = tensorify(T, A, Q)

    for t in xrange(T - 2, -1, -1):
        xsmooth[:, [t]], Vsmooth[:, :, t], VVsmooth[:, :, t + 1] = smooth_update(xsmooth[:, [t + 1]], Vsmooth[:, :, t + 1],
                                                                                 xfilt[:, [t]], Vfilt[:, :, t],
                                                                                 Vfilt[:, :, t + 1], VVfilt[:, :, t + 1],
                                                                                 A[:, :, t + 1], Q[:, :, t + 1])

    return xsmooth, Vsmooth, VVsmooth, loglik


def smooth_update(xsmooth_future, Vsmooth_future, xfilt, Vfilt, Vfilt_future, VVfilt_future, A, Q):

    # INPUTS:
    # xsmooth_future = E[X_t+1|T]
    # Vsmooth_future = Cov[X_t+1|T]
    # xfilt = E[X_t|t]
    # Vfilt = Cov[X_t|t]
    # Vfilt_future = Cov[X_t+1|t+1]
    # VVfilt_future = Cov[X_t+1,X_t|t+1]
    # A = system matrix for time t+1
    # Q = system covariance for time t+1
    #
    # OUTPUTS:
    # xsmooth = E[X_t|T]
    # Vsmooth = Cov[X_t|T]
    # VVsmooth_future = Cov[X_t+1,X_t|T]

    xpred = np.matmul(A, xfilt)
    Vpred = np.matmul(np.matmul(A, Vfilt), A.T) + Q  # Vpred = Cov[X(t+1) | t]

    J = np.matmul(np.matmul(Vfilt, A.T), np.linalg.inv(Vpred))  # smoother gain matrix
    xsmooth = xfilt + np.matmul(J, (xsmooth_future - xpred))
    Vsmooth = Vfilt + np.matmul(np.matmul(J, (Vsmooth_future - Vpred)), J.T)
    VVsmooth_future = VVfilt_future + \
        np.matmul(np.matmul((Vsmooth_future - Vfilt_future), np.linalg.inv(Vfilt_future)), VVfilt_future)

    return xsmooth, Vsmooth, VVsmooth_future


def tensorify(slices, *arrs):
    # Copy 2d matrices T times
    # and leave 3d matrices unchanged
    tensors = []
    for arr in arrs:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
            arr = np.tile(arr, (1, 1, slices))
        if arr.ndim == 3:
            assert arr.shape[2] == slices
        tensors.append(arr)
    return tensors
