"""
Additional chart patterns (8 new):
- Rising wedge / Falling wedge
- Bullish flag / Bearish flag
- Pennant
- Cup and handle
- Rounding bottom
- Gap up / Gap down
- Inside bar / Outside bar
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _find_extrema(values, distance=5):
    arr = np.asarray(values)
    peaks, _ = find_peaks(arr, distance=distance)
    troughs, _ = find_peaks(-arr, distance=distance)
    return peaks, troughs


# ============== Wedge ==============
def detect_wedges(df, lookback=30, min_r2=0.6):
    """
    Rising wedge: highs rise faster than lows (converging up). Bearish reversal.
    Falling wedge: lows fall faster than highs (converging down). Bullish reversal.
    Both narrow over time.
    """
    df = df.copy()
    n = len(df)
    df["pat_rising_wedge"] = False
    df["pat_falling_wedge"] = False
    if n < lookback:
        return df
    highs = df["high"].values
    lows = df["low"].values
    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(lookback, n):
        h = highs[i-lookback:i]
        l = lows[i-lookback:i]
        h_slope = ((x - x_mean) * (h - h.mean())).sum() / x_var
        l_slope = ((x - x_mean) * (l - l.mean())).sum() / x_var
        h_pct = h_slope / h.mean() * 100
        l_pct = l_slope / l.mean() * 100
        # R² of regression
        h_pred = h_slope * x + (h.mean() - h_slope * x_mean)
        l_pred = l_slope * x + (l.mean() - l_slope * x_mean)
        h_r2 = 1 - ((h - h_pred) ** 2).sum() / max(((h - h.mean()) ** 2).sum(), 1e-9)
        l_r2 = 1 - ((l - l_pred) ** 2).sum() / max(((l - l.mean()) ** 2).sum(), 1e-9)

        # Range narrowing: 后半段平均高低差 < 前半段
        first_half_range = (h[:lookback//2] - l[:lookback//2]).mean()
        second_half_range = (h[lookback//2:] - l[lookback//2:]).mean()
        narrowing = second_half_range < first_half_range * 0.85

        if not narrowing or h_r2 < min_r2 or l_r2 < min_r2:
            continue
        # Rising wedge: both up but highs steeper
        if h_pct > 0.05 and l_pct > 0.05 and h_pct < l_pct * 0.95:
            df.iloc[i, df.columns.get_loc("pat_rising_wedge")] = True
        # Falling wedge: both down but lows steeper
        elif h_pct < -0.05 and l_pct < -0.05 and l_pct > h_pct * 0.95:
            df.iloc[i, df.columns.get_loc("pat_falling_wedge")] = True
    return df


# ============== Flag ==============
def detect_flags(df, pole_lookback=10, flag_lookback=10):
    """
    Bull flag: strong rally (pole) then small downward consolidation (flag).
    Bear flag: strong decline (pole) then small upward consolidation (flag).
    """
    df = df.copy()
    n = len(df)
    df["pat_bull_flag"] = False
    df["pat_bear_flag"] = False
    if n < pole_lookback + flag_lookback + 5:
        return df
    closes = df["close"].values

    for i in range(pole_lookback + flag_lookback, n):
        # Pole: 之前 pole_lookback bars 涨/跌 > 5%
        pole_start = closes[i - pole_lookback - flag_lookback]
        pole_end = closes[i - flag_lookback]
        if pole_start <= 0:
            continue  # 除零保护
        pole_change = (pole_end - pole_start) / pole_start * 100

        # Flag: 最近 flag_lookback bars 反向小幅整理
        flag_start = closes[i - flag_lookback]
        flag_end = closes[i]
        if flag_start <= 0:
            continue
        flag_change = (flag_end - flag_start) / flag_start * 100
        flag_high = closes[i - flag_lookback:i+1].max()
        flag_low = closes[i - flag_lookback:i+1].min()
        if flag_low <= 0:
            continue
        flag_range = (flag_high - flag_low) / flag_low * 100

        # Bull flag
        if pole_change > 8 and -3 < flag_change < 0 and flag_range < 5:
            df.iloc[i, df.columns.get_loc("pat_bull_flag")] = True
        # Bear flag
        elif pole_change < -8 and 0 < flag_change < 3 and flag_range < 5:
            df.iloc[i, df.columns.get_loc("pat_bear_flag")] = True
    return df


# ============== Pennant ==============
def detect_pennant(df, pole_lookback=10, pennant_lookback=10):
    """
    Pennant: strong move (pole) then symmetric triangle.
    """
    df = df.copy()
    n = len(df)
    df["pat_pennant"] = False
    if n < pole_lookback + pennant_lookback + 5:
        return df
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    x = np.arange(pennant_lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(pole_lookback + pennant_lookback, n):
        # Pole: 强烈方向移动
        base_close = closes[i - pole_lookback - pennant_lookback]
        if base_close <= 0:
            continue  # 除零保护
        pole_change = (closes[i - pennant_lookback] - base_close) / base_close * 100
        if abs(pole_change) < 8:
            continue
        # Pennant: 高低逐渐收敛
        h = highs[i - pennant_lookback:i]
        l = lows[i - pennant_lookback:i]
        h_slope = ((x - x_mean) * (h - h.mean())).sum() / x_var
        l_slope = ((x - x_mean) * (l - l.mean())).sum() / x_var
        h_pct = h_slope / h.mean() * 100
        l_pct = l_slope / l.mean() * 100
        # 高斜率向下,低斜率向上 (对称三角)
        if h_pct < -0.1 and l_pct > 0.1:
            df.iloc[i, df.columns.get_loc("pat_pennant")] = True
    return df


# ============== Cup and Handle ==============
def detect_cup_handle(df, cup_lookback=60, handle_lookback=15):
    """
    Cup: U shape - decline then rounded recovery to roughly the original high
    Handle: shorter consolidation/pullback
    """
    df = df.copy()
    n = len(df)
    df["pat_cup_handle"] = False
    if n < cup_lookback + handle_lookback + 5:
        return df
    closes = df["close"].values

    for i in range(cup_lookback + handle_lookback, n):
        cup = closes[i - cup_lookback - handle_lookback:i - handle_lookback]
        handle = closes[i - handle_lookback:i]
        cup_start = cup[0]
        cup_end = cup[-1]
        cup_min = cup.min()
        if cup_start <= 0 or cup_end <= 0:
            continue  # 除零保护
        # Cup 形态: 起点和终点接近 (差<5%), 中间有显著下跌 (>10%)
        if abs(cup_start - cup_end) / cup_start > 0.05:
            continue
        cup_drawdown = (cup_min - cup_start) / cup_start * 100
        if cup_drawdown > -10:
            continue
        # Handle: 比 cup 小, 略微下跌或盘整
        handle_low = handle.min()
        handle_pullback = (handle_low - cup_end) / cup_end * 100
        if -8 < handle_pullback < 0:
            df.iloc[i, df.columns.get_loc("pat_cup_handle")] = True
    return df


# ============== Rounding bottom ==============
def detect_rounding_bottom(df, lookback=40):
    """
    Rounding bottom: smooth U shape over `lookback` bars.
    Detection: fit a quadratic, check if it's convex (positive a in ax² + bx + c).
    """
    df = df.copy()
    n = len(df)
    df["pat_rounding_bottom"] = False
    if n < lookback:
        return df
    closes = df["close"].values
    x = np.arange(lookback)

    for i in range(lookback, n):
        y = closes[i-lookback:i]
        # Fit quadratic
        try:
            a, b, c = np.polyfit(x, y, 2)
        except Exception:
            continue
        # Convex (a > 0)
        if a <= 0:
            continue
        # 启点和终点要在低点上方
        # 启点和终点要在低点上方
        if y[0] <= 0:
            continue
        y_pred = a * x ** 2 + b * x + c
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        if r2 < 0.6:
            continue
        # 起点和终点接近 (回到原位)
        if abs(y[0] - y[-1]) / y[0] > 0.05:
            continue
        # 中间有明显下沉
        if y.min() / y[0] > 0.92:
            continue
        df.iloc[i, df.columns.get_loc("pat_rounding_bottom")] = True
    return df


# ============== Gaps ==============
def detect_gaps(df, gap_threshold=0.02):
    """
    Gap up: today's open > yesterday's high * (1 + gap_threshold)
    Gap down: today's open < yesterday's low * (1 - gap_threshold)
    """
    df = df.copy()
    if len(df) < 2:
        df["pat_gap_up"] = False
        df["pat_gap_down"] = False
        return df
    df["pat_gap_up"] = df["open"] > df["high"].shift(1) * (1 + gap_threshold)
    df["pat_gap_down"] = df["open"] < df["low"].shift(1) * (1 - gap_threshold)
    return df


# ============== Inside / Outside bar ==============
def detect_inside_outside_bars(df):
    """
    Inside bar: today's high < yesterday's high AND today's low > yesterday's low
    Outside bar: today's high > yesterday's high AND today's low < yesterday's low
    """
    df = df.copy()
    if len(df) < 2:
        df["pat_inside_bar"] = False
        df["pat_outside_bar"] = False
        return df
    df["pat_inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["pat_outside_bar"] = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    return df


# ============== Master ==============
def detect_extra_patterns(df):
    df = detect_wedges(df)
    df = detect_flags(df)
    df = detect_pennant(df)
    df = detect_cup_handle(df)
    df = detect_rounding_bottom(df)
    df = detect_gaps(df)
    df = detect_inside_outside_bars(df)
    return df
