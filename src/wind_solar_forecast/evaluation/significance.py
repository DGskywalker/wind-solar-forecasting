"""Statistical significance testing: Diebold-Mariano test (with HLN correction) and Stationary Block Bootstrap."""

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DieboldMarianoResult:
    dm_statistic: float
    p_value: float
    mean_loss_diff: float
    is_significant_5pct: bool
    is_significant_1pct: bool


def diebold_mariano_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    horizon_steps: int = 24,
    alternative: str = "two-sided"
) -> DieboldMarianoResult:
    """Computes the Diebold-Mariano test statistic with Harvey, Leybourne & Newbold (1997) small-sample correction.

    Tests null hypothesis H0: E[d_t] = 0, where d_t = loss1_t - loss2_t.

    Args:
        loss1: Loss series of model 1 (e.g. absolute error or pinball loss).
        loss2: Loss series of benchmark model 2.
        horizon_steps: Forecast horizon in steps (lag truncation h - 1).
        alternative: 'two-sided', 'less' (model 1 better), or 'greater'.

    Returns:
        DieboldMarianoResult with statistic, p-value, and significance flags.
    """
    d = np.asarray(loss1) - np.asarray(loss2)
    n = len(d)
    d_mean = float(np.mean(d))

    # Compute autocovariance up to lag h - 1 using rectangular or Bartlett weights
    gamma_0 = float(np.var(d))
    autocov_sum = 0.0
    max_lag = max(horizon_steps - 1, 1)

    for k in range(1, max_lag):
        cov_k = float(np.cov(d[:-k], d[k:])[0, 1]) if len(d) > k else 0.0
        # Bartlett kernel: (1 - k / (max_lag + 1))
        weight = 1.0 - (k / (max_lag + 1.0))
        autocov_sum += 2.0 * weight * cov_k

    long_run_var = max((gamma_0 + autocov_sum) / n, 1e-9)
    dm_raw = d_mean / np.sqrt(long_run_var)

    # Harvey, Leybourne, Newbold (1997) correction factor for finite sample n and horizon h
    hln_factor = np.sqrt((n + 1.0 - 2.0 * horizon_steps + horizon_steps * (horizon_steps - 1.0) / n) / n)
    dm_stat = float(dm_raw * hln_factor)

    # Student-t distribution with n - 1 degrees of freedom
    df = max(n - 1, 1)
    if alternative == "two-sided":
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=df)))
    elif alternative == "less":
        p_val = float(stats.t.cdf(dm_stat, df=df))
    else:
        p_val = float(1.0 - stats.t.cdf(dm_stat, df=df))

    return DieboldMarianoResult(
        dm_statistic=dm_stat,
        p_value=p_val,
        mean_loss_diff=d_mean,
        is_significant_5pct=(p_val < 0.05),
        is_significant_1pct=(p_val < 0.01)
    )


def stationary_block_bootstrap(
    data: np.ndarray,
    stat_func: callable,
    n_bootstraps: int = 1000,
    mean_block_length: float = 24.0,
    random_state: int = 42
) -> tuple[float, float, tuple[float, float]]:
    """Stationary Block Bootstrap (Politis & Romano 1994) for dependent time series with geometric block lengths.

    Args:
        data: Input 1D time series data array.
        stat_func: Function mapping 1D bootstrap array to a scalar statistic.
        n_bootstraps: Number of bootstrap pseudo-samples.
        mean_block_length: Expected block length (e.g. 24 for daily periodicity).
        random_state: RNG seed.

    Returns:
        Tuple of (original_stat, bootstrap_std_err, (ci_lower_95, ci_upper_95)).
    """
    n = len(data)
    rng = np.random.default_rng(random_state)
    p_geom = 1.0 / max(mean_block_length, 1.0)

    orig_stat = float(stat_func(data))
    boot_stats = np.zeros(n_bootstraps)

    for b in range(n_bootstraps):
        boot_idx = np.zeros(n, dtype=int)
        curr_idx = rng.integers(0, n)
        for i in range(n):
            if rng.random() < p_geom:
                curr_idx = rng.integers(0, n)
            else:
                curr_idx = (curr_idx + 1) % n
            boot_idx[i] = curr_idx

        boot_stats[b] = stat_func(data[boot_idx])

    se = float(np.std(boot_stats))
    ci_low = float(np.percentile(boot_stats, 2.5))
    ci_high = float(np.percentile(boot_stats, 97.5))
    return orig_stat, se, (ci_low, ci_high)
