import numpy as np
import pandas as pd
from scipy.optimize import minimize 


def get_risk(x,cov_matrix): 
    # this function is responsible for returning variance of portfolio; in this case variance is used as risk measure
    x = np.asarray(x, float)
    return float(x.T @ cov_matrix @ x)

def mean_variance_opt(mu, cov_matrix, target_return, short_allowed=True):
    # this function performes mean-variance optimization with minimization 
    # of variance that used as risk measure and maximization of return for 
    # targeted variance
    mu = np.asarray(mu, float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, float)
    n = mu.size
    ones = np.ones(n)

    # Initial guess: equal weights
    x0 = np.full(n, 1.0 / n)

    # Constraints:
    # 1) mu^T x - target_return >= 0
    # 2) 1^T x - 1 = 0
    constraints = [
        {"type": "ineq", "fun": lambda x: float(mu @ x) - target_return},
        {"type": "eq",   "fun": lambda x: float(ones @ x) - 1.0},
    ]


    # Bounds (optional): long-only vs allow shorting
    if short_allowed:
        bounds = None
    else:
        bounds = [(0.0, 1.0)] * n

    result = minimize(
        fun=get_risk,
        x0=x0,
        args=(cov_matrix,),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 10_000}
)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x_star = result.x
    var_star = get_risk(x_star, cov_matrix)
    ret_star = float(mu @ x_star)

    return x_star, var_star, ret_star, result

def sharpe_ratio_optimization(mu, cov_matrix, rf, short_allowed=True):
    # this function performes Sharpe ratio optimization with minimization 
    # of variance that used as risk measure 
    mu = np.asarray(mu, float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, float)
    n = mu.size
    ones = np.ones(n)

    # Initial guess: equal weights
    x0 = np.full(n, 1.0 / n)

    # Constraints:
    # 1) mu - rf *x - 1 >= 0
    constraints = [
        {"type": "eq",   "fun": lambda x: float((mu - rf * ones) @ x) - 1.0},
    ]


    # Bounds (optional): long-only vs allow shorting
    if short_allowed:
        bounds = None
    else:
        bounds = [(0.0, 1.0)] * n

    result = minimize(
        fun=get_risk,
        x0=x0,
        args=(cov_matrix,),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 10_000}
)


    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x_star = result.x
    w_star = x_star / (ones @ x_star)  # normalize to sum to 1
    var_star = get_risk(w_star, cov_matrix)
    ret_star = float(mu @ w_star)

    return w_star, var_star, ret_star, result


def get_var(pnl, p):
    pnl = np.asarray(pnl)
    p = np.asarray(p)

    N = pnl.size
    sorted_losses = np.sort(pnl)

    # Compute index positions
    Kp = np.ceil((1 - p) * N).astype(int) - 1

    # Guard against out-of-bounds (numerical safety)
    Kp = np.clip(Kp, 0, N - 1)

    return sorted_losses[Kp]


def get_cvar(pnl, p):
    pnl = np.asarray(pnl)
    p_arr = np.atleast_1d(p)

    pnl_sorted = np.sort(pnl)
    N = pnl_sorted.size

    # tail sizes for each p: k = ceil((1-p)*N), at least 1, at most N
    k = np.ceil((1 - p_arr) * N).astype(int)
    k = np.clip(k, 1, N)

    # prefix sums let us get mean of first k values fast
    prefix = np.cumsum(pnl_sorted)
    cvar = prefix[k - 1] / k

    return cvar[0] if np.ndim(p) == 0 else cvar