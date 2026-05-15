"""
Advanced Pattern Detection
============================
Beyond basic candlestick patterns:
- Double bottom / Double top (W / M shapes)
- Head and shoulders / Inverse head and shoulders
- Channel detection (uptrend/downtrend channels)
- Triangle consolidation
- Breakout detection (above swing high / below swing low)
- Morning star / Evening star (3-bar patterns)

Pure detection. No signals interpreted as buy/sell.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def find_local_extrema(series, window=5):
    """Find local peaks and troughs.
    Returns (peak_indices, trough_indices).
    """
    if len(series) < 2 * window + 1:
        return np.array([]), np.array([])
    arr = np.asarray(series.values if hasattr(series, "values") else series)
    peaks, _ = find_peaks(arr, distance=window)
    troughs, _ = find_peaks(-arr, distance=window)
    return peaks, troughs


# ============== Double bottom / Double top ==============
def detect_double_patterns(df, lookback=60, tolerance=0.03):
    """
    Detect W-shape (double bottom) and M-shape (double top).
    Optimized: compute peaks/troughs once on the full series.
    """
    df = df.copy()
    df["pat_double_bottom"] = False
    df["pat_double_top"] = False

    if len(df) < lookback:
        return df

    # 一次性算所有 peaks/troughs
    closes = df["close"].values
    peaks, troughs = find_local_extrema(pd.Series(closes), window=5)
    highs = df["high"].values
    lows = df["low"].values

    # 对每个 trough(t2), 看它前 lookback bar 内是否有另一个相似深度的 trough(t1)
    # 且中间有显著的 peak
    for j in range(len(troughs)):
        t2_idx = troughs[j]
        t2_val = lows[t2_idx]
        # 找 t1: 在 [t2-lookback, t2-10] 之间的 trough
        t1_candidates = [t for t in troughs if t2_idx - lookback <= t < t2_idx - 10]
        if not t1_candidates:
            continue
        t1_idx = t1_candidates[-1]  # 最近的
        t1_val = lows[t1_idx]
        if abs(t1_val - t2_val) / max(t1_val, t2_val) >= tolerance:
            continue
        # 中间需要一个 peak
        between = [p for p in peaks if t1_idx < p < t2_idx]
        if not between:
            continue
        p_val = highs[between[0]]
        if p_val <= t1_val * 1.03:
            continue
        # 标记 t2 这一天
        df.iloc[t2_idx, df.columns.get_loc("pat_double_bottom")] = True

    # Double top 同理
    for j in range(len(peaks)):
        p2_idx = peaks[j]
        p2_val = highs[p2_idx]
        p1_candidates = [p for p in peaks if p2_idx - lookback <= p < p2_idx - 10]
        if not p1_candidates:
            continue
        p1_idx = p1_candidates[-1]
        p1_val = highs[p1_idx]
        if abs(p1_val - p2_val) / max(p1_val, p2_val) >= tolerance:
            continue
        between = [t for t in troughs if p1_idx < t < p2_idx]
        if not between:
            continue
        t_val = lows[between[0]]
        if t_val >= p1_val * 0.97:
            continue
        df.iloc[p2_idx, df.columns.get_loc("pat_double_top")] = True

    return df


# ============== Head and Shoulders ==============
def detect_head_shoulders(df, lookback=80, tolerance=0.05):
    """
    Detect head and shoulders / inverse. Optimized.
    """
    df = df.copy()
    df["pat_head_shoulders"] = False
    df["pat_inverse_head_shoulders"] = False

    if len(df) < lookback:
        return df

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    peaks, troughs = find_local_extrema(pd.Series(closes), window=4)

    # H&S: 3 peaks where middle is highest
    for j in range(2, len(peaks)):
        p1, p2, p3 = peaks[j-2], peaks[j-1], peaks[j]
        if p3 - p1 > lookback:
            continue
        v1, v2, v3 = highs[p1], highs[p2], highs[p3]
        if v2 > v1 * 1.02 and v2 > v3 * 1.02 and abs(v1 - v3) / max(v1, v3) < tolerance:
            df.iloc[p3, df.columns.get_loc("pat_head_shoulders")] = True

    # Inverse H&S
    for j in range(2, len(troughs)):
        t1, t2, t3 = troughs[j-2], troughs[j-1], troughs[j]
        if t3 - t1 > lookback:
            continue
        v1, v2, v3 = lows[t1], lows[t2], lows[t3]
        if v2 < v1 * 0.98 and v2 < v3 * 0.98 and abs(v1 - v3) / max(v1, v3) < tolerance:
            df.iloc[t3, df.columns.get_loc("pat_inverse_head_shoulders")] = True

    return df


# ============== Channel detection ==============
def detect_channels(df, lookback=40, min_r2=0.5):
    """
    Detect channels via linear regression. Optimized version.
    """
    df = df.copy()
    n = len(df)
    df["ch_slope_pct"] = np.nan
    df["ch_r2"] = np.nan
    df["pat_uptrend_channel"] = False
    df["pat_downtrend_channel"] = False
    df["pat_horizontal_channel"] = False

    if n < lookback:
        return df

    closes = df["close"].values
    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    slope_pcts = np.full(n, np.nan)
    r2s = np.full(n, np.nan)

    for i in range(lookback, n):
        window = closes[i-lookback:i]
        y_mean = window.mean()
        cov = ((x - x_mean) * (window - y_mean)).sum()
        slope = cov / x_var
        intercept = y_mean - slope * x_mean
        y_pred = slope * x + intercept
        ss_res = ((window - y_pred) ** 2).sum()
        ss_tot = ((window - y_mean) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        slope_pct = slope / y_mean * 100
        slope_pcts[i] = slope_pct
        r2s[i] = r2

    df["ch_slope_pct"] = slope_pcts
    df["ch_r2"] = r2s

    high_r2 = r2s > min_r2
    df["pat_uptrend_channel"] = high_r2 & (slope_pcts > 0.05)
    df["pat_downtrend_channel"] = high_r2 & (slope_pcts < -0.05)
    # 注: horizontal_channel 不再使用 - 历史样本太少(n<30) 无法可靠回测
    # 保留列以兼容,但永远 False
    df["pat_horizontal_channel"] = False

    return df


# ============== Triangle consolidation ==============
def detect_triangles(df, lookback=30):
    """Triangle detection. Optimized."""
    df = df.copy()
    n = len(df)
    df["pat_triangle_symmetric"] = False
    df["pat_triangle_ascending"] = False
    df["pat_triangle_descending"] = False

    if n < lookback:
        return df

    highs = df["high"].values
    lows = df["low"].values
    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    sym = np.zeros(n, dtype=bool)
    asc = np.zeros(n, dtype=bool)
    desc = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        h_win = highs[i-lookback:i]
        l_win = lows[i-lookback:i]
        h_slope = ((x - x_mean) * (h_win - h_win.mean())).sum() / x_var
        l_slope = ((x - x_mean) * (l_win - l_win.mean())).sum() / x_var
        h_pct = h_slope / h_win.mean() * 100
        l_pct = l_slope / l_win.mean() * 100

        if h_pct < -0.03 and l_pct > 0.03:
            sym[i] = True
        elif abs(h_pct) < 0.03 and l_pct > 0.03:
            asc[i] = True
        elif h_pct < -0.03 and abs(l_pct) < 0.03:
            desc[i] = True

    df["pat_triangle_symmetric"] = sym
    df["pat_triangle_ascending"] = asc
    df["pat_triangle_descending"] = desc
    return df


# ============== Breakout detection ==============
def detect_breakouts(df, lookback=20):
    """
    Breakout: today's close exceeds the highest close of past N days
              (or falls below the lowest close).

    Returns:
      pat_breakout_up: today's close > max(close[t-lookback:t-1])
      pat_breakout_down: today's close < min(close[t-lookback:t-1])
    """
    df = df.copy()
    if len(df) < lookback + 1:
        df["pat_breakout_up"] = False
        df["pat_breakout_down"] = False
        return df
    rolling_max = df["close"].rolling(lookback).max().shift(1)  # past N excluding today
    rolling_min = df["close"].rolling(lookback).min().shift(1)
    df["pat_breakout_up"] = df["close"] > rolling_max
    df["pat_breakout_down"] = df["close"] < rolling_min
    return df


# ============== Morning Star / Evening Star ==============
def detect_star_patterns(df):
    """
    Morning star: bear bar + small body (gap down) + bull bar (closes above mid of bar 1)
    Evening star: bull bar + small body (gap up) + bear bar (closes below mid of bar 1)
    """
    df = df.copy()
    if len(df) < 3:
        df["pat_morning_star"] = False
        df["pat_evening_star"] = False
        return df

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    body = np.abs(c - o)
    avg_body = pd.Series(body).rolling(20).mean().values

    n = len(df)
    morning = np.zeros(n, dtype=bool)
    evening = np.zeros(n, dtype=bool)

    for i in range(2, n):
        if np.isnan(avg_body[i]):
            continue
        # Bar 1 (i-2)
        bar1_bear = c[i-2] < o[i-2]
        bar1_big = body[i-2] > avg_body[i] * 1.0
        bar1_mid = (o[i-2] + c[i-2]) / 2
        # Bar 2 (i-1) - small body
        bar2_small = body[i-1] < avg_body[i] * 0.5
        # Bar 3 (i)
        bar3_bull = c[i] > o[i]
        bar3_big = body[i] > avg_body[i] * 0.8
        bar3_above_mid = c[i] > bar1_mid

        # Morning star
        if bar1_bear and bar1_big and bar2_small and bar3_bull and bar3_big and bar3_above_mid:
            morning[i] = True

        # Evening star
        bar1_bull = c[i-2] > o[i-2]
        bar3_bear = c[i] < o[i]
        bar3_below_mid = c[i] < bar1_mid

        if bar1_bull and bar1_big and bar2_small and bar3_bear and bar3_big and bar3_below_mid:
            evening[i] = True

    df["pat_morning_star"] = morning
    df["pat_evening_star"] = evening
    return df


# ============== Master ==============
def detect_all_advanced_patterns(df):
    df = detect_double_patterns(df)
    df = detect_head_shoulders(df)
    df = detect_channels(df)
    df = detect_triangles(df)
    df = detect_breakouts(df)
    df = detect_star_patterns(df)
    return df
