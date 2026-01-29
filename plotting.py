import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optimizer import mean_variance_opt, sharpe_ratio_optimization 


def plot_efficient_frontier(front_returns, mu, cov_matrix, rf):
    """
    front_returns: (K,) array of target returns
    mu: (n,) expected returns
    cov_matrix: (n,n) covariance matrix
    minimizer_for_target_return: function(target_return) -> weights (n,)
        (You can also accept mu,cov_matrix; adjust the call below accordingly.)
    """
    front_returns = np.asarray(front_returns, float)
    mu = np.asarray(mu, float).reshape(-1)
    cov_matrix = np.asarray(cov_matrix, float)

    K = front_returns.size
    n = mu.size

    # Preallocate for speed
    W = np.empty((K, n), dtype=float)
    vols = np.full(K, np.nan, dtype=float)
    rets = np.full(K, np.nan, dtype=float)

    # Solve for each target return
    for k, r_t in enumerate(front_returns):
        x_star, var_star, ret_star, result = mean_variance_opt(mu, cov_matrix, r_t, short_allowed=True)
        w = np.asarray(x_star, float).reshape(-1)

        # store weights (optional but useful)
        W[k, :] = w

        # stats
        rets[k] = float(mu @ w)
        vols[k] = float(np.sqrt(w @ cov_matrix @ w) * 100)

    w_sharpe, var_sharpe, return_sharpe, result_sharpe = sharpe_ratio_optimization(mu, cov_matrix, rf, short_allowed=True)
    vol_sharpe = np.sqrt(var_sharpe)*100

    # In case some targets are infeasible and returned NaNs
    ok = np.isfinite(vols) & np.isfinite(rets)

    plt.figure()
    plt.plot(vols[ok], rets[ok], linestyle='-', marker='.', markersize=2, label = "Efficient Frontier")
    plt.scatter(vol_sharpe, return_sharpe, marker = "X", color = "m", label="Max Sharpe ratio", s = 55)
    plt.xlabel("Volatility %")
    plt.ylabel("Expected return")
    plt.title("Mean-Variance optimized Portfolio")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(corr, labels=None):
    corr = np.asarray(corr, float)

    plt.figure()
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)

    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()

def plot_return_violins(returns, asset_names=None):
    returns = np.asarray(returns, float)

    T, N = returns.shape

    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(N)]

    # Convert to long format
    df = pd.DataFrame(returns, columns=asset_names)
    df_long = df.melt(var_name="Asset", value_name="Return")

    plt.figure()
    sns.violinplot(
        data=df_long,
        x="Asset",
        y="Return",
        inner="quartile",
        cut=0
    )

    plt.xticks(rotation=90)
    plt.title("Return distributions")
    plt.ylabel("Returns")
    plt.tight_layout()
    plt.show()

def plot_var_cvar(pnl, var, cvar, conf_level, nbins):
    plt.hist(pnl, bins = nbins )
    plt.title("Distribution of portfolio's profit and loss (P&L) ")
    plt.xlabel("Portfolio P&L")
    label_var = rf"$VaR_{{{int(conf_level*100)}\%}}$"
    label_cvar = rf"$CVaR_{{{int(conf_level*100)}\%}}$"
    plt.axvline(x=var, color='g', label=label_var)
    plt.axvline(x=cvar, color='r', label=label_cvar)
    plt.grid(True)
    plt.legend()