"""
Split Conformal Prediction
============================
为任意点预测器附加分布无关的预测区间, 提供边缘覆盖保证 (marginal coverage)。

核心思想:
  1. 用训练集拟合任意预测器 f
  2. 在独立校准集上计算非一致性分数 s_i = |y_i - f(x_i)|
  3. 计算分位数 q = quantile(s_1..s_n, ceil((n+1)(1-α))/n)
  4. 新预测的 (1-α) 区间 = [f(x_new) - q, f(x_new) + q]

理论保证:
  P(y_new ∈ 区间) ≥ 1 - α  (假设数据可交换 exchangeable)

实际局限:
  - 时间序列违反 exchangeability (有自相关), 真实覆盖率可能略低于声明
  - 区间宽度反映历史不确定性, 不预知 regime change
  - 输出 P(return > 0) 是基于校准残差的经验概率, 不是真实未来概率

模块设计:
  - SplitConformal: 通用一维回归 conformal 包装
  - probability_forecast: 端到端 baseline AR(5) + 滚动 conformal 校准
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


# =============================================================================
# Generic Split Conformal Wrapper
# =============================================================================
class SplitConformal:
    """
    给定 (predictions, actuals) 对, 计算非一致性分数, 提供
    .predict_interval(y_hat, alpha) 返回 [low, high]。

    使用流程:
        sc = SplitConformal()
        sc.fit(cal_predictions, cal_actuals)  # 校准集
        low, high = sc.predict_interval(y_hat=0.5, alpha=0.05)  # 95% CI
        p_up = sc.prob_above(y_hat=0.5, threshold=0.0)  # P(实际值 > 0)
    """

    def __init__(self):
        self.scores_ = None  # 排序后的非一致性分数
        self.residuals_ = None  # 有符号残差 (用于经验概率)
        self.n_ = 0

    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> "SplitConformal":
        """根据校准集拟合."""
        pred = np.asarray(predictions, dtype=float).ravel()
        act = np.asarray(actuals, dtype=float).ravel()
        if pred.shape != act.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {act.shape}")
        mask = np.isfinite(pred) & np.isfinite(act)
        pred, act = pred[mask], act[mask]
        if len(pred) < 10:
            raise ValueError(f"need >=10 calibration points, got {len(pred)}")

        residuals = act - pred  # 有符号
        self.residuals_ = residuals
        self.scores_ = np.sort(np.abs(residuals))
        self.n_ = len(residuals)
        return self

    def predict_interval(self, y_hat: float | np.ndarray, alpha: float = 0.1):
        """
        返回 (1-alpha) 预测区间。
        返回单值: alpha=0.1 → 90% CI; alpha=0.05 → 95% CI。
        """
        if self.scores_ is None:
            raise RuntimeError("call .fit(...) first")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        # 经典 split conformal 修正:
        # quantile_level = ceil((n+1)*(1-alpha)) / n, 上限 1.0
        n = self.n_
        q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        q = np.quantile(self.scores_, q_level)
        if isinstance(y_hat, np.ndarray):
            return y_hat - q, y_hat + q
        return float(y_hat - q), float(y_hat + q)

    def prob_above(self, y_hat: float, threshold: float = 0.0) -> float:
        """
        基于校准残差的经验分布, 估计 P(y_actual > threshold)。
        模型: y_actual ≈ y_hat + ε, ε 来自校准残差经验分布。
        所以 P(y_actual > thr) = P(ε > thr - y_hat) = (residuals > thr - y_hat).mean()
        """
        if self.residuals_ is None:
            raise RuntimeError("call .fit(...) first")
        delta = threshold - y_hat
        return float((self.residuals_ > delta).mean())

    def coverage_check(self, alpha: float = 0.1) -> float:
        """
        在校准集上自检覆盖率 (应 ≈ 1-alpha, 反之说明残差分布不平稳)。
        """
        if self.scores_ is None:
            raise RuntimeError("call .fit(...) first")
        n = self.n_
        q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        q = np.quantile(self.scores_, q_level)
        within = (np.abs(self.residuals_) <= q).mean()
        return float(within)


# =============================================================================
# Baseline forecaster: AR(p) on log returns + sum to N-day horizon
# =============================================================================
def _fit_ar_logreturns(log_returns: np.ndarray, p: int = 5):
    """
    最简单的 AR(p) on log returns, 用最小二乘解 (含截距)。
    返回 (coefs, intercept), 长度 p+1。
    coefs[0] = lag-1 系数, coefs[p-1] = lag-p 系数。
    """
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < p + 30:
        return None
    # 构造设计矩阵
    n = len(r) - p
    X = np.zeros((n, p))
    for i in range(p):
        X[:, i] = r[p - 1 - i: p - 1 - i + n]
    y = r[p:]
    X_aug = np.column_stack([np.ones(n), X])
    # 用 lstsq 解
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept = coef[0]
    ar = coef[1:]
    return ar, intercept


def _ar_forecast(history: np.ndarray, ar: np.ndarray, intercept: float, horizon: int) -> float:
    """
    Recursive AR forecast: 输出 horizon 步累计 log return (= sum of step forecasts)。
    """
    p = len(ar)
    h = list(history[-p:])  # 最近 p 个真实值, 反向 (lag-1, lag-2, ...)
    h = h[::-1]
    cumsum = 0.0
    for _ in range(horizon):
        # h: [r_{t-1}, r_{t-2}, ..., r_{t-p}]
        next_r = intercept + sum(ar[i] * h[i] for i in range(p))
        cumsum += next_r
        # 更新历史: 把 next_r 放在头部
        h = [next_r] + h[:-1]
    return float(cumsum)


# =============================================================================
# End-to-end: probability forecast with rolling conformal calibration
# =============================================================================
def probability_forecast(
    df: pd.DataFrame,
    horizon: int = 10,
    alpha: float = 0.1,
    lookback: int = 504,
    cal_size: int = 100,
    ar_order: int = 5,
) -> dict:
    """
    端到端: 在最近 lookback 根 bar 上滚动训练 AR(p), 在最后 cal_size 根上做 conformal 校准,
    然后对最后一根 bar 的下 horizon 天预测 + 区间 + 上涨概率。

    Parameters
    ----------
    df : DataFrame, 必须含 'close' 列, 时间升序
    horizon : 预测期 (天), 例如 10 = "10 个交易日后"
    alpha : 1-alpha 是名义覆盖率, alpha=0.1 → 90% CI
    lookback : 用于训练 + 校准的总历史天数
    cal_size : 校准集大小 (从 lookback 末尾切出)
    ar_order : AR 阶数

    Returns
    -------
    dict with keys:
      ok, error, point_pct, lower_pct, upper_pct, p_up, p_up_5pct, p_dn_5pct,
      calibration_n, calibration_coverage, horizon, alpha, method
    """
    if df is None or df.empty:
        return {"ok": False, "error": "empty df"}
    if "close" not in df.columns:
        return {"ok": False, "error": "missing close column"}

    close = df["close"].astype(float).values
    if len(close) < lookback + horizon + ar_order + 5:
        return {"ok": False, "error": f"need >={lookback + horizon + ar_order + 5} bars, got {len(close)}"}

    log_close = np.log(close)
    log_ret_1d = np.diff(log_close)  # 长度 = len(close) - 1

    # 切片: 用最后 lookback 个 1-day log return
    if len(log_ret_1d) < lookback + horizon:
        return {"ok": False, "error": "insufficient history for lookback+horizon"}

    window = log_ret_1d[-lookback:]
    train_size = lookback - cal_size - horizon
    if train_size < ar_order + 30:
        return {"ok": False, "error": "insufficient train segment"}

    # 1. 在 train segment 上拟合 AR
    train_ret = window[:train_size]
    fit = _fit_ar_logreturns(train_ret, p=ar_order)
    if fit is None:
        return {"ok": False, "error": "AR fit failed"}
    ar, intercept = fit

    # 2. 校准: 在 cal_size 个不同起点上做 horizon-步预测, 比对真实
    cal_segment = window[train_size: train_size + cal_size + horizon]
    if len(cal_segment) < cal_size + horizon:
        return {"ok": False, "error": "cal_segment too short"}

    preds, actuals = [], []
    for i in range(cal_size):
        # 用 i 之前的全部数据做历史
        hist = np.concatenate([train_ret, cal_segment[:i]])
        if len(hist) < ar_order:
            continue
        pred_h = _ar_forecast(hist, ar, intercept, horizon)
        # 真实未来 horizon 天的累计 log return
        actual_h = float(cal_segment[i: i + horizon].sum())
        preds.append(pred_h)
        actuals.append(actual_h)

    if len(preds) < 30:
        return {"ok": False, "error": "calibration set too small after assembly"}

    sc = SplitConformal().fit(np.array(preds), np.array(actuals))

    # 3. 在最新一根 bar 做点预测
    full_hist = np.concatenate([train_ret, cal_segment])  # 等于 window
    pred_h = _ar_forecast(full_hist, ar, intercept, horizon)

    # 4. 区间 (log return) → 转回百分比 return
    low_log, high_log = sc.predict_interval(pred_h, alpha=alpha)
    point_pct = (math.exp(pred_h) - 1) * 100
    lower_pct = (math.exp(low_log) - 1) * 100
    upper_pct = (math.exp(high_log) - 1) * 100

    # 5. 概率: P(实际累计 log return > 0) → P(price 上涨)
    p_up = sc.prob_above(pred_h, threshold=0.0)
    # 也算 P(>+5%) 和 P(<-5%) 用于风险参考
    log_5 = math.log(1.05)
    log_neg5 = math.log(0.95)
    p_up_5pct = sc.prob_above(pred_h, threshold=log_5)
    p_dn_5pct = 1.0 - sc.prob_above(pred_h, threshold=log_neg5)

    return {
        "ok": True,
        "error": None,
        "horizon": horizon,
        "alpha": alpha,
        "method": f"AR({ar_order}) on log returns + split conformal",
        "point_pct": round(point_pct, 2),
        "lower_pct": round(lower_pct, 2),
        "upper_pct": round(upper_pct, 2),
        "p_up": round(p_up, 3),
        "p_up_5pct": round(p_up_5pct, 3),
        "p_dn_5pct": round(p_dn_5pct, 3),
        "calibration_n": len(preds),
        "calibration_coverage": round(sc.coverage_check(alpha), 3),
        # baseline (无信息基准): 简单 P(过去 cal_segment 中 horizon 天上涨的频率)
        "naive_p_up_baseline": round(float((np.array(actuals) > 0).mean()), 3),
    }


# =============================================================================
# Adapter: 给任意外部点预测器加 conformal (供 Task 2 Kronos 使用)
# =============================================================================
def conformalize_external_forecaster(
    df: pd.DataFrame,
    forecaster_fn,
    horizon: int = 10,
    alpha: float = 0.1,
    cal_size: int = 100,
) -> dict:
    """
    给一个外部预测器 (例如 Kronos / Chronos) 加 conformal 区间。

    Parameters
    ----------
    df : DataFrame, 必须含 'close', 时间升序
    forecaster_fn : callable
        签名: forecaster_fn(close_history: np.ndarray, horizon: int) -> float
        返回: horizon 期累计 log return 的点预测
    horizon : int
    alpha : float
    cal_size : 校准窗口大小 (在最后 cal_size+horizon 根 bar 上做)

    Returns
    -------
    dict (同 probability_forecast 结构, method 字段反映外部模型)
    """
    if df is None or df.empty or "close" not in df.columns:
        return {"ok": False, "error": "bad df"}
    close = df["close"].astype(float).values
    if len(close) < cal_size + horizon + 30:
        return {"ok": False, "error": "insufficient history"}

    log_close = np.log(close)

    preds, actuals = [], []
    # 校准: 滚动 cal_size 个起点, 每个起点用截至该点的 close 预测 horizon 后
    for i in range(cal_size):
        end_idx = len(close) - cal_size - horizon + i  # 切到这里做预测
        if end_idx < 60:
            continue
        hist = close[:end_idx]
        try:
            pred_log_h = float(forecaster_fn(hist, horizon))
        except Exception:
            continue
        actual_log_h = float(log_close[end_idx + horizon - 1] - log_close[end_idx - 1])
        preds.append(pred_log_h)
        actuals.append(actual_log_h)

    if len(preds) < 20:
        return {"ok": False, "error": "calibration failed; forecaster errored or too few points"}

    sc = SplitConformal().fit(np.array(preds), np.array(actuals))

    # 最新预测
    try:
        pred_h = float(forecaster_fn(close, horizon))
    except Exception as e:
        return {"ok": False, "error": f"forecaster failed at production call: {e}"}

    low_log, high_log = sc.predict_interval(pred_h, alpha=alpha)
    point_pct = (math.exp(pred_h) - 1) * 100
    lower_pct = (math.exp(low_log) - 1) * 100
    upper_pct = (math.exp(high_log) - 1) * 100
    p_up = sc.prob_above(pred_h, threshold=0.0)
    log_5 = math.log(1.05)
    log_neg5 = math.log(0.95)
    p_up_5pct = sc.prob_above(pred_h, threshold=log_5)
    p_dn_5pct = 1.0 - sc.prob_above(pred_h, threshold=log_neg5)

    return {
        "ok": True,
        "error": None,
        "horizon": horizon,
        "alpha": alpha,
        "method": "external forecaster + split conformal",
        "point_pct": round(point_pct, 2),
        "lower_pct": round(lower_pct, 2),
        "upper_pct": round(upper_pct, 2),
        "p_up": round(p_up, 3),
        "p_up_5pct": round(p_up_5pct, 3),
        "p_dn_5pct": round(p_dn_5pct, 3),
        "calibration_n": len(preds),
        "calibration_coverage": round(sc.coverage_check(alpha), 3),
        "naive_p_up_baseline": round(float((np.array(actuals) > 0).mean()), 3),
    }
