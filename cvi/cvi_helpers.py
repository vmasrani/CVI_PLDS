import numpy as np
from utils import column_vec, chol_sample
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
        elbo_k += 0.5 * np.log(np.linalg.det(Rk))

        elbo += elbo_k

    elbo += logZ
    return elbo.item()

# -----------------------------
# Overwrite with own likelihood
# ----------------------------
def non_conjugate_likelihood(*args):
    # Init
    y, mc_latent, C = args
    ls, sample_size, T = mc_latent.shape
    os, T = y.shape

    # For numerical stability
    eps = 1e-6

    mc_latent = np.swapaxes(mc_latent, 2, 1)
    ybroad = np.tile(np.expand_dims(y, axis=2), [1, 1, sample_size])

    # Initalize tensorflow variables
    C = tf.Variable(C, name='C', dtype=tf.float32)

    # Initalize tensorflow placeholders
    X = tf.placeholder("float",  shape=mc_latent.shape)
    Y = tf.placeholder("float", shape=(os, T, sample_size))

    Z = tf.tensordot(C, X, axes = [[1], [0]])

    rate     = tf.nn.softplus(Z) + eps
    pdf      = tf.contrib.distributions.Poisson(rate=rate, allow_nan_stats=False)
    logpdf   = pdf.log_prob(value=Y)
    feeddict = {X: mc_latent, Y: ybroad}

    # Return logpdf, mean parameter, and feeddict
    return logpdf, Z, feeddict


def E_log_p_mc(y, mc_latent, C):
    logpdf, Z, feeddict = non_conjugate_likelihood(y, mc_latent, C)

    f   = logpdf
    df  = tf.gradients(f, Z)
    d2f = tf.gradients(df, Z)

    # Run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        f_results = sess.run(f, feeddict)
        df_results = sess.run(df, feeddict)[0]
        d2f_results = sess.run(d2f, feeddict)[0]

    # Average MC samples
    f = f_results.mean(axis=2)
    # First derivative
    gm = df_results.mean(axis=2)
    # Second derivative
    gv = d2f_results.mean(axis=2) / 2.0

    return f, gm, gv


def make_y_R_tilde(tlam_1, tlam_2):
    assert tlam_1.shape == tlam_2.shape
    os, T = tlam_1.shape

    var_tilde = 1. / (-2 * tlam_2)
    y_tilde = np.divide(tlam_1, -2 * tlam_2)
    R_tilde = np.zeros((os, os, T))

    for i in range(T):
        np.fill_diagonal(R_tilde[:, :, i], var_tilde[:, i])

    return y_tilde, R_tilde


def sample_posterior(x, V, nSamples):
    ls, T = x.shape
    raw_noise = np.random.randn(ls, nSamples)
    samples = np.zeros((ls, nSamples, T))
    for t in range(T):
        samples[:, :, t] = chol_sample(raw_noise, x[:, t], V[:, :, t])
    return samples

