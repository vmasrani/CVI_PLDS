import numpy as np


def learn_kalman(data, A, C, Q, R, initx, initV, smoother=None, max_iter=10, verbose=True, **kwargs):
    thresh = 1e-3
    if data.ndim == 2:
        data = np.expand_dims(data, 3)

    N = data.shape[2]
    ss = A.shape[0]
    os = C.shape[0]

    alpha = np.zeros([os, os])
    Tsum = 0

    for ex in range(N):
        y = data[:, :, ex]
        T = y.shape[1]
        Tsum = Tsum + T
        alpha_temp = np.zeros([os, os])
        for t in range(T):
            alpha_temp = alpha_temp + np.outer(y[:, t], y[:, t])
        alpha = alpha + alpha_temp

    previous_loglik = -np.inf
    loglik = 0
    converged = 0
    num_iter = 1
    LL = []

    while (num_iter <= max_iter) and not converged:
        # ===========E step============
        delta  = np.zeros([os, ss])
        gamma  = np.zeros([ss, ss])
        gamma1 = np.zeros([ss, ss])
        gamma2 = np.zeros([ss, ss])
        beta   = np.zeros([ss, ss])
        P1sum  = np.zeros([ss, ss])
        x1sum  = np.zeros([ss, 1])
        loglik = 0

        for ex in range(N):
            y = data[:, :, ex]
            T = y.shape[1]
            beta_t, gamma_t, delta_t, gamma1_t, gamma2_t, x1, V1, loglik_t = Estep(y, A, C, Q, R, initx, initV, smoother=smoother)
            beta   = beta   + beta_t
            gamma  = gamma  + gamma_t
            delta  = delta  + delta_t
            gamma1 = gamma1 + gamma1_t
            gamma2 = gamma2 + gamma2_t
            P1sum  = P1sum  + V1 + np.outer(x1, x1)
            x1sum  = x1sum  + x1
            loglik = loglik + loglik_t

        LL.append(loglik)

        if verbose:
            print "iteration %d, loglik = %f" % (num_iter, loglik)

        num_iter += 1
        # -----------end E step------------

        # ===========M step============
        Tsum1 = Tsum - N

        # Save old values to reset if user wants to fix a matrix
        A_old = np.array(A)
        Q_old = np.array(Q)
        C_old = np.array(C)
        R_old = np.array(R)
        initx_old = np.array(initx)
        initV_old = np.array(initV)

        #                                              # ----------Matlab------------
        A = np.matmul(beta, np.linalg.inv(gamma1))     # A = beta * inv(gamma1);
        Q = (gamma2 - np.matmul(A, beta.T)) / Tsum1    # Q = (gamma2 - A*beta') / Tsum1;
        C = np.matmul(delta, np.linalg.inv(gamma))     # C = delta * inv(gamma);
        R = (alpha - np.matmul(C, delta.T)) / Tsum     # R = (alpha - C*delta') / Tsum;

        initx = x1sum / N
        initV = P1sum / N - np.outer(initx, initx)

        # Fix matrices
        # (This is not optimized because we compute the entire matrix and throw away values)
        for key in kwargs:
            # Fix entire matrix
            if key == 'fix_A' and kwargs[key] == 'full':
                A = A_old
            if key == 'fix_Q' and kwargs[key] == 'full':
                Q = Q_old
            if key == 'fix_C' and kwargs[key] == 'full':
                C = C_old
            if key == 'fix_R' and kwargs[key] == 'full':
                R = R_old
            if key == 'fix_initx' and kwargs[key] == 'full':
                initx = initx_old
            if key == 'fix_initV' and kwargs[key] == 'full':
                initV = initV_old

            # Learn only diagonal elements
            if key == 'fix_A' and kwargs[key] == 'diag':
                np.fill_diagonal(A_old, A.diagonal())
                A = A_old
            if key == 'fix_Q' and kwargs[key] == 'diag':
                np.fill_diagonal(Q_old, Q.diagonal())
                Q = Q_old
            if key == 'fix_C' and kwargs[key] == 'diag':
                np.fill_diagonal(C_old, C.diagonal())
                C = C_old
            if key == 'fix_R' and kwargs[key] == 'diag':
                np.fill_diagonal(R_old, R.diagonal())
                R = R_old

        converged = em_converged(loglik, previous_loglik, thresh)
        previous_loglik = loglik
        # -----------end of M step------------
    # -------end of while loop--------

    return A, C, Q, R, initx, initV, LL


def Estep(y, A, C, Q, R, initx, initV, smoother=None):
    # Compute the (expected) sufficient statistics for a single Kalman filter sequence.

    os, T = y.shape
    ss = A.shape[0]

    if isinstance(smoother, list):
        [xsmooth, Vsmooth, VVsmooth, loglik] = smoother
    else:
        [xsmooth, Vsmooth, VVsmooth, loglik] = smoother(y, A, C, Q, R, initx, initV)

    # Fix transpose bug
    VVsmooth = np.transpose(VVsmooth, (1, 0, 2))

    delta = np.zeros([os, ss])
    gamma = np.zeros([ss, ss])
    beta  = np.zeros([ss, ss])

    for t in range(T):
        delta = delta + np.outer(y[:, t], xsmooth[:, t])
        gamma = gamma + np.outer(xsmooth[:, t], xsmooth[:, t]) + Vsmooth[:, :, t]
        if t > 0:
            beta = beta + np.outer(xsmooth[:, t], xsmooth[:, t - 1]) + VVsmooth[:, :, t]

    gamma1 = gamma - np.outer(xsmooth[:, T - 1], xsmooth[:, T - 1]) - Vsmooth[:, :, T - 1]
    gamma2 = gamma - np.outer(xsmooth[:, 0], xsmooth[:, 0]) - Vsmooth[:, :, 0]

    x1 = np.array([xsmooth[:, 0]]).T
    V1 = Vsmooth[:, :, 0]

    return beta, gamma, delta, gamma1, gamma2, x1, V1, loglik


def em_converged(loglik, previous_loglik, thresh=1e-3, check_increased=True):
    eps = np.finfo(float).eps
    inf = float('inf')
    converged = False
    if check_increased:
        if loglik - previous_loglik < -1e-3:  # allow for a little imprecision
            print '******likelihood decreased from %6.4f to %6.4f!\n' % (previous_loglik, loglik)

    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + eps) / 2

    if (delta_loglik == inf) & (avg_loglik == inf):
        return converged

    if (delta_loglik / avg_loglik) < thresh:
        converged = True

    return converged
