"""
tvp_ffbs_gibbs.py

Time-varying-parameter (TVP) factor model estimated via Carter & Kohn (1994)
Forward-Filtering Backward-Sampling (FFBS), embedded in a Metropolis-within-
Gibbs sampler. NOT executed yet -- this is for review of the MH step before
we run anything on real data.

Model
-----
State equation (random walk betas, one per factor, k = 14 factors):
    beta_t = beta_{t-1} + eta_t,      eta_t ~ N(0, Q),      Q = diag(tau_1^2, ..., tau_k^2)

Observation equation (single asset's excess return regressed on k factors):
    y_t = X_t' beta_t + eps_t,        eps_t ~ N(0, sigma^2)

Priors
------
    beta_0 ~ N(b0, P0)                     -- seeded from the static hierarchical
                                               posterior (asset-specific beta mean/var,
                                               NOT the cross-sectional tau_j from the
                                               static model -- see conversation notes)
    tau_j  ~ HalfStudentT(nu, scale_j)     -- per factor, NON-conjugate -> MH step
    sigma^2 ~ InverseGamma(a0, b0_obs)     -- conjugate -> direct Gibbs draw

One Gibbs sweep
----------------
    1. Kalman filter forward pass given current {tau_j}, sigma^2
       (explicit Riccati recursion -- see kalman_filter_forward)
    2. Backward sample the whole beta_{0:T} path (Carter & Kohn backward recursion)
    3. MH-within-Gibbs update of each tau_j, conditional on the sampled beta path
    4. Conjugate Gibbs draw of sigma^2, conditional on the sampled beta path
    5. Record draw, repeat
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# ---------------------------------------------------------------------------
# 0. Half-Student-t log-density, built from a PyMC distribution object so the
#    prior evaluation is consistent with however you specify tau_j's prior
#    elsewhere in the model (nu, scale). Compiled once to a fast numeric
#    function via pytensor so the MH step below isn't rebuilding a graph on
#    every proposal.
# ---------------------------------------------------------------------------
def make_half_student_t_logpdf(nu: float, scale: float):
    x = pt.scalar("x")
    dist = pm.HalfStudentT.dist(nu=nu, sigma=scale)
    logp_expr = pm.logp(dist, x)
    f = pm.pytensorf.compile([x], logp_expr)
    return f  # f(x) -> log p(x) under HalfStudentT(nu, scale), x > 0


# ---------------------------------------------------------------------------
# 1. Kalman filter forward pass -- Riccati equations written out explicitly,
#    not hidden inside a library call.
# ---------------------------------------------------------------------------
def kalman_filter_forward(y, X, b0, P0, Q, sigma2):
    """
    y: (T,) observed excess returns for this asset
    X: (T, k) factor design matrix (rows = X_t')
    b0: (k,) prior mean of beta_0
    P0: (k, k) prior covariance of beta_0
    Q:  (k, k) state innovation covariance, diag(tau_1^2, ..., tau_k^2)
    sigma2: scalar observation noise variance

    Returns filtered means/covariances at every t, AND the predicted
    (pre-update) means/covariances -- both are needed by the backward pass.
    """
    T, k = X.shape

    b_pred = np.zeros((T, k))          # beta_{t|t-1}
    P_pred = np.zeros((T, k, k))       # P_{t|t-1}
    b_filt = np.zeros((T, k))          # beta_{t|t}
    P_filt = np.zeros((T, k, k))       # P_{t|t}

    b_prev, P_prev = b0, P0

    for t in range(T):
        # --- Predict step (Riccati time-update) ---
        b_t_pred = b_prev                       # random walk: E[beta_t | I_{t-1}] = beta_{t-1|t-1}
        P_t_pred = P_prev + Q                   # Riccati: uncertainty grows by Q each step

        # --- Update step (Riccati measurement-update) ---
        x_t = X[t, :]                            # (k,)
        S_t = x_t @ P_t_pred @ x_t.T + sigma2     # scalar innovation variance
        K_t = (P_t_pred @ x_t) / S_t              # (k,) Kalman gain

        innovation = y[t] - x_t @ b_t_pred        # scalar prediction error
        b_t_filt = b_t_pred + K_t * innovation
        # Joseph form for numerical stability (keeps P_t_filt symmetric/PSD):
        I_k = np.eye(k)
        P_t_filt = (I_k - np.outer(K_t, x_t)) @ P_t_pred @ (I_k - np.outer(K_t, x_t)).T \
                   + np.outer(K_t, K_t) * sigma2

        b_pred[t], P_pred[t] = b_t_pred, P_t_pred
        b_filt[t], P_filt[t] = b_t_filt, P_t_filt
        b_prev, P_prev = b_t_filt, P_t_filt

    return b_pred, P_pred, b_filt, P_filt


# ---------------------------------------------------------------------------
# 2. Carter & Kohn backward sampling -- draws the FULL beta_{0:T} path jointly,
#    not just point estimates. This is the "S" (simulation smoother) step.
# ---------------------------------------------------------------------------
def ffbs_backward_sample(b_pred, P_pred, b_filt, P_filt, Q, rng):
    T, k = b_filt.shape
    beta_draw = np.zeros((T, k))

    # t = T-1: sample directly from the final filtered distribution
    beta_draw[T - 1] = rng.multivariate_normal(b_filt[T - 1], P_filt[T - 1])

    # t = T-2, ..., 0: sample beta_t | beta_{t+1}, y_{1:t}
    for t in range(T - 2, -1, -1):
        # Smoothing gain (Riccati-derived, same structure as the KF gain
        # but linking beta_t to beta_{t+1} instead of to an observation):
        J_t = P_filt[t] @ np.linalg.inv(P_pred[t + 1])
        mean_t = b_filt[t] + J_t @ (beta_draw[t + 1] - b_pred[t + 1])
        cov_t = P_filt[t] - J_t @ P_pred[t + 1] @ J_t.T
        cov_t = 0.5 * (cov_t + cov_t.T)  # symmetrize, guards against float drift

        beta_draw[t] = rng.multivariate_normal(mean_t, cov_t)

    return beta_draw


# ---------------------------------------------------------------------------
# 3. Metropolis-within-Gibbs update for tau_j -- THE STEP TO REVIEW.
#
#    Full conditional target (up to normalizing constant):
#        p(tau_j | beta_path) ~  [ product_t Normal(eta_{t,j}; 0, tau_j^2) ]  *  HalfStudentT(tau_j; nu, scale)
#    where eta_{t,j} = beta_{t,j} - beta_{t-1,j} are the sampled state
#    innovations for factor j, already fixed from step 2 above.
#
#    Proposal: random walk on log(tau_j), symmetric in log-space.
#        log(tau_j') = log(tau_j) + N(0, step^2)
#    Because the proposal is symmetric in log(tau_j) but the target density
#    is written in terms of tau_j, the acceptance ratio needs the Jacobian
#    d(tau)/d(log tau) = tau -- see the extra tau_prop / tau_curr factor below.
# ---------------------------------------------------------------------------
def mh_update_tau_j(tau_curr, eta_j, half_t_logpdf, step_size, rng):
    """
    tau_curr: current scalar value of tau_j
    eta_j: (T,) array of sampled state innovations for factor j (beta_t - beta_{t-1})
    half_t_logpdf: compiled function from make_half_student_t_logpdf(nu, scale_j)
    step_size: RW step size on the log scale (tune for ~0.44 acceptance rate)
    """
    # --- Propose on the log scale ---
    log_tau_curr = np.log(tau_curr)
    log_tau_prop = log_tau_curr + rng.normal(0, step_size)
    tau_prop = np.exp(log_tau_prop)

    # --- Log target density at current and proposed values ---
    def log_target(tau):
        n = eta_j.shape[0]
        loglik = -n * np.log(tau) - 0.5 * np.sum(eta_j ** 2) / (tau ** 2)
        logprior = half_t_logpdf(tau)
        return loglik + logprior

    log_target_curr = log_target(tau_curr)
    log_target_prop = log_target(tau_prop)

    # --- Jacobian-corrected log acceptance ratio ---
    # log[ p(tau') * tau' / (p(tau) * tau) ]
    log_accept_ratio = (log_target_prop + np.log(tau_prop)) - (log_target_curr + np.log(tau_curr))

    if np.log(rng.uniform()) < log_accept_ratio:
        return tau_prop, True   # accepted
    else:
        return tau_curr, False  # rejected, keep current value


# ---------------------------------------------------------------------------
# 4. Conjugate Gibbs draw for sigma^2 (Inverse-Gamma, standard normal-linear
#    conjugacy -- no MH needed here).
# ---------------------------------------------------------------------------
def gibbs_draw_sigma2(y, X, beta_path, a0, b0_obs, rng):
    T = y.shape[0]
    resid = y - np.einsum("tk,tk->t", X, beta_path)  # y_t - X_t' beta_t, all t
    a_post = a0 + T / 2
    b_post = b0_obs + 0.5 * np.sum(resid ** 2)
    # InverseGamma(a_post, b_post) via 1 / Gamma(a_post, scale=1/b_post)
    sigma2_draw = 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)
    return sigma2_draw


# ---------------------------------------------------------------------------
# 5. One full Gibbs sweep, tying the pieces above together.
# ---------------------------------------------------------------------------
def one_gibbs_sweep(y, X, b0, P0, tau, sigma2, half_t_logpdfs, mh_step_sizes,
                     sigma2_prior, rng):
    """
    tau: (k,) current tau_j values
    half_t_logpdfs: list of k compiled logpdf functions, one per factor
    mh_step_sizes: (k,) RW step sizes on log(tau_j), one per factor
    sigma2_prior: (a0, b0_obs) for the Inverse-Gamma prior on sigma^2
    """
    k = X.shape[1]
    Q = np.diag(tau ** 2)

    # --- 1. Kalman filter forward (Riccati recursion) ---
    b_pred, P_pred, b_filt, P_filt = kalman_filter_forward(y, X, b0, P0, Q, sigma2)

    # --- 2. Backward sample the full beta path ---
    beta_path = ffbs_backward_sample(b_pred, P_pred, b_filt, P_filt, Q, rng)

    # --- 3. MH-within-Gibbs update for each tau_j ---
    eta = np.diff(beta_path, axis=0, prepend=beta_path[[0]] - b0)  # state innovations, (T, k)
    tau_new = tau.copy()
    accepted = np.zeros(k, dtype=bool)
    for j in range(k):
        tau_new[j], accepted[j] = mh_update_tau_j(
            tau[j], eta[:, j], half_t_logpdfs[j], mh_step_sizes[j], rng
        )

    # --- 4. Conjugate draw for sigma^2 ---
    a0, b0_obs = sigma2_prior
    sigma2_new = gibbs_draw_sigma2(y, X, beta_path, a0, b0_obs, rng)

    return beta_path, tau_new, sigma2_new, accepted


if __name__ == "__main__":
    print("Module defined -- not executing the sampler yet, per review request.")
    print("Review focus: mh_update_tau_j() -- the log-scale RW proposal and its Jacobian-corrected acceptance ratio.")
