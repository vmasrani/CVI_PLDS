import numpy as np
from utils import check_format, column_vec, chol_sample
import tensorflow as tf

# \begin{align*}
# L(q_k) &= \mathbb{E}_{q_k}\bigg[\log\bigg(\frac{p(y,x)}{q(x)}\bigg)\bigg]\\
# q(x_k)&\sim \mathcal{N}(\hat{u}_{k|k},P_{k|k}) \\
# \\
# L(q_t) &= \mathbb{E}_{q_t}\bigg[\log\bigg(\frac{\prod_{k=1}^{K}P_{\theta}(y_{k}|Cx_k)\mathcal{N}_{\theta}(x_k|Ax_{k-1},Q)\mathcal{N}_{\theta}(x_0)}{\frac{1}{Z_{\theta}}\prod_{k=1}^{K}\mathcal{N}_{\theta}(\tilde{y}_{k,t}|Cx_k,\tilde{R}_{k,t})\mathcal{N}_{\theta}(x_k|Ax_{k-1},Q)\mathcal{N}_{\theta}(x_0)}\bigg)\bigg]
# \\
# L(q_t) &= \sum_{k=1}^{K}\mathbb{E}_{q_t}[\log P_{\theta}(y_{k}|Cx_k)] - \mathbb{E}_{q_t}[\log \mathcal{N}_{\theta}(\tilde{y}_{k,t}|Cx_k,\tilde{R}_{k,t}) + \log Z_t(\theta)
# \\
# L(q_t) &\approx \sum_{k=1}^{K}\bigg[ \frac{1}{S}\sum_{s=1}^{S}[\log P_{\theta}(y_{k}|Cx_k^s)]\bigg] +\frac{1}{2}(\tilde{y}_{k,t} - C\hat{u}_{k,t})^T\tilde{R}_{k,t}^{-1}(\tilde{y}_{k,t} - C\hat{u}_{k,t}) + \frac{1}{2}Tr(C^T\tilde{R}^{-1}CP_k) + \frac{1}{2}\log{|\tilde{R}|} + \log_{\theta}P(Y_{1:K})
# \end{align*}


def get_elbo(Elog_p, ytilde, xfilt, Vfilt, R, C, logZ):
    os, K = ytilde.shape
    elbo = 0
    for k in range(K):
        # v = y_kt - C*u_kt
        v = column_vec(ytilde[:, k] - np.matmul(C, xfilt[:, k]))
        Rk = R[:, :, k]
        Rinv = np.linalg.inv(Rk)

        elbo_k = np.sum(Elog_p[:, k])
        elbo_k += 0.5 * np.linalg.multi_dot([v.T, Rinv, v])
        elbo_k += 0.5 * \
            np.trace(np.linalg.multi_dot([C.T, Rinv, C, Vfilt[:, :, k]]))
        # Potential issue here
        elbo_k += 0.5 * np.log(np.linalg.det(Rk))

        elbo += elbo_k

    elbo += logZ
    return elbo.item()


def E_log_p_mc(y, mc_latent, C, D, sample_size=10):
    ls, sample_size, T = mc_latent.shape
    os, T = y.shape

    eps = 1e-6

    mc_latent = np.swapaxes(mc_latent, 2, 1)
    ybroad = np.tile(np.expand_dims(y, axis=2), [1, 1, sample_size])

    sess = tf.InteractiveSession()

    # Initalize tensorflow variables
    C = tf.Variable(C, name='C', dtype=tf.float32)
    D = tf.Variable(D, name='D', dtype=tf.float32)

    # Initalize tensorflow placeholders
    X = tf.placeholder("float",  shape=mc_latent.shape)
    Y = tf.placeholder("float", shape=(os, T, sample_size))

    Z = tf.tensordot(C, X, axes = [[1], [0]])
    Z = tf.add(Z, tf.expand_dims(D, axis=2))

    # exp link
    # pdf = tf.contrib.distributions.Poisson(log_rate=Z, allow_nan_stats=False)

    # softplus
    rate = tf.nn.softplus(Z) + eps
    pdf = tf.contrib.distributions.Poisson(rate=rate, allow_nan_stats=False)

    f   = pdf.log_prob(value=Y)
    df  = tf.gradients(f, Z)
    d2f = tf.gradients(df, Z)

    # Run
    sess.run(tf.global_variables_initializer())
    feeddict = {X: mc_latent, Y: ybroad}

    f_results = sess.run(f, feeddict)
    df_results = sess.run(df, feeddict)[0]
    d2f_results = sess.run(d2f, feeddict)[0]

    # Average MC samples
    f = f_results.mean(axis=2)
    gm = df_results.mean(axis=2)
    gv = d2f_results.mean(axis=2) / 2.0
    sess.close()
    return f, gm, gv


def maximize_non_conjugate(data, lr = 0.01, iters=500, verbose=True):
    ls, sample_size, T = data["Z"].shape
    os, _ = data["Y"].shape

    sess = tf.InteractiveSession()
    # Change indexing to conform to TensorFlow broadcast rules
    # mc_samples = np.reshape(data["Z"], (ls, T, sample_size))
    eps = 1e-6
    mc_samples = np.swapaxes(data["Z"], 2, 1)

    # Initalize tensorflow variables
    C = tf.Variable(data["C"], name='C', dtype=tf.float32)
    D = tf.Variable(data["D"], name='D', dtype=tf.float32)

    X = tf.placeholder("float",  shape=mc_samples.shape)
    Y = tf.placeholder("float", shape=(os, T, sample_size))

    Z = tf.tensordot(C, X, axes = [[1], [0]])
    Z = tf.add(Z, tf.expand_dims(D, axis=2))

    # exp link
    # pdf = tf.contrib.distributions.Poisson(log_rate=Z, allow_nan_stats=False)

    # softplus
    rate = tf.nn.softplus(Z) + eps
    pdf = tf.contrib.distributions.Poisson(rate=rate, allow_nan_stats=False)

    cost = tf.reduce_sum(tf.reduce_mean(pdf.log_prob(value=Y), axis=2))

    # logpdf = tf.multiply(Y, log_rate) - tf.exp(log_rate) - tf.lgamma(Y + 1)
    # cost = tf.reduce_sum(tf.reduce_mean(logpdf, axis=2))

    ybroad = np.tile(np.expand_dims(data['Y'], axis=2), [1, 1, sample_size])

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # Adam Optimizer
    opt_op = optimizer.minimize(-cost)
    feeddict = {X: mc_samples, Y: ybroad}

    sess.run(tf.global_variables_initializer())

    for epoch in range(iters):
        _, loss_val = sess.run([opt_op, cost], feeddict)
        if verbose:
            print "Epoch: %04d, train_loss: %04f" % (epoch + 1,loss_val )

    Cval, Dval = C.eval(), D.eval()
    sess.close()

    return Cval, Dval




def make_y_R_tilde(tlam_1, tlam_2):
    assert tlam_1.shape == tlam_2.shape
    os, T = tlam_1.shape

    var_tilde = 1. / (-2 * tlam_2)
    y_tilde = np.divide(tlam_1, -2 * tlam_2)
    R_tilde = np.zeros((os, os, T))

    for i in range(T):
        np.fill_diagonal(R_tilde[:, :, i], var_tilde[:, i])

    return y_tilde, R_tilde


# def kalman_backwards_sampler(x, V, A, Q, sample_size=10):
#     # Implementation follows:
#     # http://people.isy.liu.se/rt/schon/Publications/LindstenS2013.pdf
#     #  - section 1.7 Backward Simulation in Linear Gaussian SSMs,
#     #  - equations 1.18b, 1.18c
#     ls, T = x.shape
#     raw_noise = np.random.randn(ls, sample_size)

#     samples = np.zeros((ls, sample_size, T))

#     # Initialize
#     samples[:, :, T - 1] = chol_sample(raw_noise, x[:, T - 1], V[:, :, T - 1])

#     for t in xrange(T - 2, -1, -1):
#         Pt = V[:, :, t]
#         xt = x[:, t]
#         xt = np.expand_dims(x[:, t], -1)

#         # Make:
#         # ut = xt + P*A.T*(Q + APA.T)^-1*(x_(t+1) - A*xt)
#         # Mt = Pt - Pt*A.T*(Q + APA.T)^-1*APt

#         # Dummy variables:
#         # B = x_(t+1) - A*xt
#         # Z = Q + APA.T
#         # L = P*A.T

#         # ut = xt + L*(Z^-1*B)
#         B = samples[:, :, t + 1] - np.matmul(A, xt)
#         Z = Q + np.matmul(A, np.matmul(Pt, A.T))
#         L = np.matmul(Pt, A.T)
#         ut = xt + np.matmul(L, np.linalg.solve(Z, B))

#         # Mt = Pt - L*(Z^-1*L.T)
#         Mt = Pt - np.matmul(L, np.linalg.solve(Z, L.T))

#         # Sample
#         samples[:, :, t] = chol_sample(raw_noise, ut, Mt)

#     return samples


def sample_posterior(x, V, nSamples):
    ls, T = x.shape
    raw_noise = np.random.randn(ls, nSamples)
    samples = np.zeros((ls, nSamples, T))
    for t in range(T):
        samples[:, :, t] = chol_sample(raw_noise, x[:, t], V[:, :, t])

    return samples

