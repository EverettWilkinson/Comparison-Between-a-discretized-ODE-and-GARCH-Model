import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf


def compute_log_returns(prices):
    return np.diff(np.log(prices))


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def garch_variance_path(params, returns):
    omega, alpha, beta = params
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)   # unconditional variance as seed
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def garch_neg_ll(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    sigma2 = garch_variance_path(params, returns)
    if np.any(sigma2 <= 0):
        return 1e10
    # sum from t=1; sigma2[0] is only the seed
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2[1:]) + returns[1:] ** 2 / sigma2[1:])


def fit_garch(returns):
    # Optimise on percent returns to avoid gradient underflow in omega.
    # sigma²_pct = 1e4·sigma², so omega_pct / 1e4 = omega_original.
    # alpha and beta are scale-invariant.
    r_pct = returns * 100
    init_omega_pct = float(np.var(r_pct) * (1 - 0.1 - 0.8))
    result = minimize(
        garch_neg_ll,
        np.array([init_omega_pct, 0.1, 0.8]),
        args=(r_pct,),
        method="L-BFGS-B",
        bounds=[(1e-8, None), (1e-8, 1), (1e-8, 1)],
    )
    omega_pct, alpha, beta = result.x
    params = np.array([omega_pct / 1e4, alpha, beta])
    return params, result


def garch_oos_path(params, r_in, r_oos):
    omega, alpha, beta = params
    sigma2_in = garch_variance_path(params, r_in)
    last_v, last_eps2 = sigma2_in[-1], r_in[-1] ** 2
    sigma2 = np.zeros(len(r_oos))
    for t in range(len(r_oos)):
        sigma2[t] = omega + alpha * last_eps2 + beta * last_v
        last_eps2 = r_oos[t] ** 2
        last_v    = sigma2[t]
    return sigma2


# ── Difference-Equation Model  v_t = v_{t-1} + k(v*−v_{t-1}) + c·r²_{t-1} ──

def de_variance_path(params, returns):
    k, v_star, c = params
    T = len(returns)
    v = np.zeros(T)
    v[0] = v_star                            # unconditional level as seed
    for t in range(1, T):
        v[t] = v[t - 1] + k * (v_star - v[t - 1]) + c * returns[t - 1] ** 2
    return v


def de_neg_ll(params, returns):
    k, v_star, c = params
    # recursion v_t = (1−k)v_{t-1} + k·v* + c·r²_{t-1}; stable iff 0 < k < 2
    if not (0 < k < 2) or v_star <= 0 or c < 0:
        return 1e10
    v = de_variance_path(params, returns)
    if np.any(v <= 0):
        return 1e10
    # sum from t=1; v[0] is only the seed
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(v[1:]) + returns[1:] ** 2 / v[1:])


def fit_de(returns):
    init_v_star = float(np.var(returns))
    result = minimize(
        de_neg_ll,
        np.array([0.05, init_v_star, 0.03]),
        args=(returns,),
        method="L-BFGS-B",
        bounds=[(1e-6, 2 - 1e-6), (1e-8, None), (1e-8, None)],
    )
    return result.x, result


def de_oos_path(params, r_in, r_oos):
    k, v_star, c = params
    v_in = de_variance_path(params, r_in)
    last_v, last_eps2 = v_in[-1], r_in[-1] ** 2
    v = np.zeros(len(r_oos))
    for t in range(len(r_oos)):
        v[t]      = last_v + k * (v_star - last_v) + c * last_eps2
        last_eps2 = r_oos[t] ** 2
        last_v    = v[t]
    return v


# ── Evaluation ────────────────────────────────────────────────────────────────

def point_losses(forecast, realized):
    e = forecast - realized
    return {
        "MSE":   e ** 2,
        "RMSE":  e ** 2,
        "MAE":   np.abs(e),
        "QLIKE": np.log(forecast) + realized / forecast,
    }


def summarise_errors(losses):
    return {
        "MSE":   float(np.mean(losses["MSE"])),
        "RMSE":  float(np.sqrt(np.mean(losses["RMSE"]))),
        "MAE":   float(np.mean(losses["MAE"])),
        "QLIKE": float(np.mean(losses["QLIKE"])),
    }


def diebold_mariano(loss1, loss2, lags=1):
    """DM test with Newey-West HAC s.e.  Positive stat → model 1 loses more."""
    d     = loss1 - loss2
    T     = len(d)
    d_bar = np.mean(d)
    lrv   = np.var(d, ddof=1)
    for h in range(1, lags + 1):
        gamma_h = np.mean((d[h:] - d_bar) * (d[:-h] - d_bar))
        lrv    += 2 * (1 - h / (lags + 1)) * gamma_h
    dm_stat = d_bar / np.sqrt(max(lrv, 1e-30) / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ticker = "SPY"
    data   = yf.download(ticker, start="2015-01-01", end="2024-12-31",
                         auto_adjust=True, progress=False)
    prices  = data["Close"].values.flatten()
    returns = compute_log_returns(prices)

    split = int(len(returns) * 0.8)
    r_in, r_oos  = returns[:split], returns[split:]
    realized_var = r_oos ** 2

    print(f"{ticker}  |  in-sample: {len(r_in)}  out-of-sample: {len(r_oos)}\n")

    # ── Fit ──
    garch_params, garch_res = fit_garch(r_in)
    de_params,    de_res    = fit_de(r_in)

    garch_ll = -garch_neg_ll(garch_params, r_in)
    de_ll    = -de_neg_ll(de_params, r_in)

    omega, alpha, beta = garch_params
    k, v_star, c       = de_params

    W = 62
    print("=" * W)
    print("  IN-SAMPLE")
    print("=" * W)

    print(f"\nGARCH(1,1)")
    print(f"  omega          = {omega:.4e}")
    print(f"  alpha          = {alpha:.6f}")
    print(f"  beta           = {beta:.6f}")
    print(f"  alpha+beta     = {alpha + beta:.6f}")
    print(f"  uncond. vol    = {np.sqrt(omega / (1 - alpha - beta) * 252):.4f}  (ann.)")
    print(f"  log-likelihood = {garch_ll:.4f}   converged={garch_res.success}")

    print(f"\nDifference Equation")
    print(f"  k              = {k:.6f}")
    print(f"  v*             = {v_star:.4e}")
    print(f"  c              = {c:.6f}")
    print(f"  log-likelihood = {de_ll:.4f}   converged={de_res.success}")

    # ── OOS forecasts ──
    garch_oos = garch_oos_path(garch_params, r_in, r_oos)
    de_oos    = de_oos_path(de_params, r_in, r_oos)

    rho_oos = float(np.corrcoef(garch_oos, de_oos)[0, 1])

    g_losses = point_losses(garch_oos, realized_var)
    d_losses = point_losses(de_oos,    realized_var)

    g_err = summarise_errors(g_losses)
    d_err = summarise_errors(d_losses)

    dm_mse,   p_mse   = diebold_mariano(g_losses["MSE"],   d_losses["MSE"])
    dm_qlike, p_qlike = diebold_mariano(g_losses["QLIKE"], d_losses["QLIKE"])

    print(f"\n{'=' * W}")
    print(f"  OUT-OF-SAMPLE  (proxy: squared returns)")
    print(f"{'=' * W}")
    print(f"\n  ρ(GARCH forecast, DE forecast) = {rho_oos:.6f}\n")
    print(f"{'Metric':<8}  {'GARCH(1,1)':>14}  {'Diff Eq':>14}")
    print("-" * 40)
    for m in ("MSE", "RMSE", "MAE", "QLIKE"):
        print(f"{m:<8}  {g_err[m]:>14.4e}  {d_err[m]:>14.4e}")

    print(f"\nDiebold-Mariano  (+ stat → GARCH loses more than Diff Eq)")
    print(f"  MSE   loss: DM = {dm_mse:>+8.4f},  p = {p_mse:.4f}")
    print(f"  QLIKE loss: DM = {dm_qlike:>+8.4f},  p = {p_qlike:.4f}")

    # ── Plots ──
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(returns, color="steelblue", linewidth=0.5)
    axes[0].axvline(split, color="red", linestyle="--", linewidth=1, label="Train / Test split")
    axes[0].set_title(f"{ticker} — Log Returns")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    sigma2_in = garch_variance_path(garch_params, r_in)
    v_in      = de_variance_path(de_params, r_in)
    axes[1].plot(np.sqrt(sigma2_in * 252), color="darkorange",     lw=0.8, label="GARCH(1,1)")
    axes[1].plot(np.sqrt(v_in      * 252), color="mediumseagreen", lw=0.8, ls="--", label="Diff Eq")
    axes[1].set_title("In-Sample Conditional Volatility (annualized)")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    x = np.arange(split, split + len(r_oos))
    axes[2].plot(x, realized_var, color="steelblue",      lw=0.5, alpha=0.55,
                 label="Squared returns (realized proxy)")
    axes[2].plot(x, garch_oos,   color="darkorange",     lw=1.0, label="GARCH(1,1)")
    axes[2].plot(x, de_oos,      color="mediumseagreen", lw=1.0, ls="--", label="Diff Eq")
    axes[2].set_title(f"Out-of-Sample Forecasts  |  ρ(GARCH, DE) = {rho_oos:.4f}")
    axes[2].set_ylabel("Variance")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
