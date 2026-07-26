# Volatility Clustering via a GARCH-Adjacent ODE

A continuous-time model of equity return volatility, written as an ordinary differential equation, benchmarked against the standard discrete-time GARCH(1,1) model on daily S&P 500 returns.

This is the code for an MTH 323 (differential equations modeling) project. The question it answers: when you discretize a simple mean-reverting ODE for the variance, do you recover GARCH(1,1), and does it forecast volatility just as well?

## The idea

Equity volatility clusters — large moves follow large moves, calm follows calm. The discrete-time benchmark for this is GARCH(1,1):

```
σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
```

This project instead starts from a continuous-time ODE for the instantaneous variance `v(t)`:

```
dv/dt = k(v* − v) + c·r(t)²
```

where `v*` is the long-run variance, `k` is the mean-reversion speed, and `c` scales the effect of the squared return. The squared return enters as a forcing term built from observed data, so once the returns are given the equation is deterministic. This is a deliberate simplification of the stochastic-volatility and diffusion-limit models (Nelson 1990, Heston 1993), which keep a second random noise term that this model drops.

Discretizing the ODE with a forward Euler step at a daily interval gives

```
vₜ = (1 − k)·vₜ₋₁ + k·v* + c·r²ₜ₋₁
```

which is algebraically a GARCH(1,1) recursion. The correspondence is exact term by term: `β ↔ 1 − k`, `ω ↔ k·v*`, and `α ↔ c`.

## What the code does

- Downloads daily S&P 500 (SPY) prices for 2015–2024 via `yfinance` and computes log returns.
- Splits the series 80/20 into in-sample and out-of-sample windows.
- Fits both models by maximum likelihood under a conditionally normal return assumption, so the two are estimated in identical likelihood frameworks and any difference in fit reflects the model, not the fitting procedure. GARCH is optimized on percent-scaled returns to avoid gradient underflow in `ω`.
- Rolls one-step-ahead variance forecasts over the out-of-sample window with fixed parameters.
- Compares the two on in-sample log-likelihood and on out-of-sample MSE, RMSE, MAE, and QLIKE against a squared-return proxy for the unobservable true variance, with a Diebold–Mariano test for whether the accuracy difference is significant.
- Plots the returns with the train/test split, the in-sample conditional volatility from both models, and the out-of-sample forecasts against the realized proxy.

## Result

The two models come out nearly identical, which is what the term-by-term equivalence predicts. On the 2015–2024 SPY sample:

- In-sample log-likelihoods differ by a fraction of a point out of ~6600.
- The fitted parameters map onto each other as expected: `β ≈ 1 − k`, `α ≈ c`, and `k·v* ≈ ω` to the printed digits.
- The two forecast paths correlate at ρ ≈ 0.9999.
- GARCH holds a consistent out-of-sample edge that the Diebold–Mariano test flags as statistically significant, but the magnitude of the difference is well under one percent and economically negligible. With thousands of observations the DM test has enough power to detect a trivial gap, so the significance should be read as "consistent," not "large."

The takeaway is that the essential mechanism behind clustering is mean reversion driven by squared returns; the extra stochastic structure in the full continuous-time models buys almost nothing for point forecasting.

## Requirements

- Python 3.9+
- `numpy`
- `scipy`
- `matplotlib`
- `yfinance`

```bash
pip install numpy scipy matplotlib yfinance
```

## Usage

```bash
python volatility_ode.py
```

The script prints the in-sample parameter estimates and log-likelihoods, the out-of-sample loss table and Diebold–Mariano statistics, and then shows the three diagnostic plots. To use a different ticker or date range, edit `ticker`, `start`, and `end` in `main()`.

## Notes

- The squared daily return is a noisy but unbiased proxy for the true variance; it is correct on average but very noisy day to day, so out-of-sample losses are averaged over the full window.
- QLIKE and MSE are the two loss functions that rank volatility forecasts correctly under a noisy proxy (Patton 2011). The QLIKE reported here uses the ranking-equivalent shortcut form.
- The ODE recursion is stable for `0 < k < 2`; the fitted `k` sits well inside that range.
