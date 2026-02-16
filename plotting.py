import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from optimizer import mean_variance_opt, sharpe_ratio_optimization, get_var_gauss, get_cvar_gauss 


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


def plot_var_es_conf_level(conf_levels, var, es, title, ax=None):
    """Plot VaR and ES vs confidence level on a given axis (or create one)."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(conf_levels, var, label="VaR")
    ax.plot(conf_levels, es, label="ES")
    ax.set_title(title)
    ax.set_xlabel("Confidence level")
    ax.grid(True)
    ax.legend()

    return ax

def gaussian_fit_curve(pnl, n_points):
    pnl = np.asarray(pnl)
    x = np.linspace(pnl.min(), pnl.max(), n_points)
    mu = pnl.mean()
    sigma = pnl.std(ddof=1)  # sample std (often preferred)
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    return x, pdf, mu, sigma


def plot_pnl_with_gaussian(
    pnl,
    title,
    conf_level,
    nbins,
    n_points,
    var_hist=None,
    cvar_hist=None,
):
    pnl = np.asarray(pnl)

    # Gaussian fit
    x = np.linspace(pnl.min(), pnl.max(), n_points)
    mu = pnl.mean()
    sigma = pnl.std(ddof=1)
    pdf = norm.pdf(x, loc=mu, scale=sigma)

    var_gauss  = get_var_gauss(mu, sigma, conf_level)
    cvar_gauss = get_cvar_gauss(mu, sigma, conf_level)

    label_var  = rf"$VaR_{{{int(conf_level*100)}\%}}$"
    label_cvar = rf"$CVaR_{{{int(conf_level*100)}\%}}$"

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

    # --- 1) Distribution only ---
    axes[0].hist(pnl, bins=nbins, density=True, alpha=0.4, label="historical")
    axes[0].plot(x, pdf, label="Gaussian")
    axes[0].set_title("Distribution")
    axes[0].set_xlabel("Portfolio P&L")
    axes[0].grid(True)
    axes[0].legend()

    # --- 2) VaR ---
    axes[1].hist(pnl, bins=nbins, density=True, alpha=0.4)
    axes[1].plot(x, pdf)
    if var_hist is not None:
        axes[1].axvline(var_hist, color="magenta", label=label_var + " historic")
    axes[1].axvline(var_gauss, color="r", label=label_var + " param")
    axes[1].set_title("VaR")
    axes[1].set_xlabel("Portfolio P&L")
    axes[1].grid(True)
    axes[1].legend()

    # --- 3) CVaR ---
    axes[2].hist(pnl, bins=nbins, density=True, alpha=0.4)
    axes[2].plot(x, pdf)
    if cvar_hist is not None:
        axes[2].axvline(cvar_hist, color="magenta", label=label_cvar + " historic")
    axes[2].axvline(cvar_gauss, color="r", label=label_cvar + " param")
    axes[2].set_title("CVaR")
    axes[2].set_xlabel("Portfolio P&L")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_risk_param_vs_hist(conf_levels, hist, param, title, ylabel, param_label, hist_label, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(conf_levels, param, label=param_label, color="red")
    ax.plot(conf_levels, hist,  label=hist_label,  color="magenta")
    ax.set_title(title)
    ax.set_xlabel("Confidence level")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    return ax