"""
Extended indicator set for multi-signal study.
- 5 types of cross signals (MACD, MA, KDJ, DMI, PSAR)
- 4 types of S/R proximity flags (200MA, 60d swing, pivot, BB)
- 5 types of candle pattern flags
"""

import numpy as np
import pandas as pd


# ============== Cross Indicators ==============
def add_macd_cross(df, fast=12, slow=26, signal=9):
    """MACD金叉死叉"""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_sig"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_state"] = np.sign(df["macd_hist"]).fillna(0).astype(int)
    df["macd_gc"] = (df["macd_state"] == 1) & (df["macd_state"].shift(1) <= 0)
    df["macd_dc"] = (df["macd_state"] == -1) & (df["macd_state"].shift(1) >= 0)
    df.loc[df.index[:48], ["macd_gc", "macd_dc", "macd_state"]] = [False, False, 0]
    return df


def add_ma_cross(df):
    """均线金叉死叉: 5/20 (短), 20/50 (中), 50/200 (长)"""
    df = df.copy()
    for p in [5, 20, 50, 200]:
        df[f"sma{p}"] = df["close"].rolling(p).mean()
    # 短期金叉: 5 上穿 20
    short_above = df["sma5"] > df["sma20"]
    df["ma_short_gc"] = short_above & ~short_above.shift(1).fillna(False).astype(bool)
    df["ma_short_dc"] = ~short_above & short_above.shift(1).fillna(False).astype(bool)
    # 中期金叉: 20 上穿 50
    mid_above = df["sma20"] > df["sma50"]
    df["ma_mid_gc"] = mid_above & ~mid_above.shift(1).fillna(False).astype(bool)
    df["ma_mid_dc"] = ~mid_above & mid_above.shift(1).fillna(False).astype(bool)
    # 长期金叉: 50 上穿 200 (经典 Golden Cross)
    long_above = df["sma50"] > df["sma200"]
    df["ma_long_gc"] = long_above & ~long_above.shift(1).fillna(False).astype(bool)
    df["ma_long_dc"] = ~long_above & long_above.shift(1).fillna(False).astype(bool)
    df.loc[df.index[:201], ["ma_short_gc", "ma_short_dc", "ma_mid_gc", "ma_mid_dc",
                   "ma_long_gc", "ma_long_dc"]] = False
    return df


def add_kdj(df, n=9, m=3):
    """KDJ - K线穿越D线"""
    df = df.copy()
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = ((df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)) * 100
    # K = 累积EMA, D = K的EMA
    df["kdj_k"] = rsv.ewm(span=m, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(span=m, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    above = df["kdj_k"] > df["kdj_d"]
    df["kdj_gc"] = above & ~above.shift(1).fillna(False).astype(bool)
    df["kdj_dc"] = ~above & above.shift(1).fillna(False).astype(bool)
    # 极值条件: 金叉在超卖区(K<20)更可信; 死叉在超买区(K>80)更可信
    df["kdj_gc_oversold"] = df["kdj_gc"] & (df["kdj_k"] < 30)
    df["kdj_dc_overbought"] = df["kdj_dc"] & (df["kdj_k"] > 70)
    df.loc[df.index[:31], ["kdj_gc", "kdj_dc", "kdj_gc_oversold", "kdj_dc_overbought"]] = False
    return df


def add_dmi(df, n=14):
    """DMI/ADX - +DI 穿越 -DI"""
    df = df.copy()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.rolling(n).mean()
    above = plus_di > minus_di
    df["dmi_gc"] = above & ~above.shift(1).fillna(False).astype(bool)
    df["dmi_dc"] = ~above & above.shift(1).fillna(False).astype(bool)
    # 强趋势条件: ADX > 25 时金叉/死叉更可靠
    df["dmi_gc_strong"] = df["dmi_gc"] & (df["adx"] > 25)
    df["dmi_dc_strong"] = df["dmi_dc"] & (df["adx"] > 25)
    df.loc[df.index[:31], ["dmi_gc", "dmi_dc", "dmi_gc_strong", "dmi_dc_strong"]] = False
    return df


def add_psar(df, af=0.02, af_max=0.2):
    """Parabolic SAR - 翻转点"""
    df = df.copy()
    n = len(df)
    psar = df["close"].copy().to_numpy().astype(float)  # 用 to_numpy 替代 .values
    psar = np.array(psar, copy=True)  # 确保独立的可写数组
    is_long = np.ones(n, dtype=bool)
    af_arr = np.full(n, af)
    ep = df["high"].iloc[0] if not df.empty else 0

    for i in range(1, n):
        prev_psar = psar[i-1]
        prev_ep = ep
        prev_af = af_arr[i-1]
        if is_long[i-1]:
            new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            if df["low"].iloc[i] < new_psar:
                # Flip to short
                is_long[i] = False
                psar[i] = prev_ep
                ep = df["low"].iloc[i]
                af_arr[i] = af
            else:
                is_long[i] = True
                psar[i] = new_psar
                if df["high"].iloc[i] > prev_ep:
                    ep = df["high"].iloc[i]
                    af_arr[i] = min(prev_af + af, af_max)
                else:
                    ep = prev_ep
                    af_arr[i] = prev_af
        else:
            new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            if df["high"].iloc[i] > new_psar:
                is_long[i] = True
                psar[i] = prev_ep
                ep = df["high"].iloc[i]
                af_arr[i] = af
            else:
                is_long[i] = False
                psar[i] = new_psar
                if df["low"].iloc[i] < prev_ep:
                    ep = df["low"].iloc[i]
                    af_arr[i] = min(prev_af + af, af_max)
                else:
                    ep = prev_ep
                    af_arr[i] = prev_af
    df["psar"] = psar
    df["psar_long"] = is_long
    # PSAR 翻转点: 用 numpy bitwise(对 bool dtype 安全), 避免 ~bool 弃用警告
    psar_long_prev = pd.Series(is_long).shift(1).fillna(False).astype(bool).to_numpy().astype(bool)
    df["psar_flip_up"] = is_long & (~psar_long_prev)  # 两边都是 ndarray bool, 安全
    df["psar_flip_dn"] = (~is_long) & psar_long_prev
    return df


# ============== Support / Resistance ==============
def add_sr_proximity(df, lookback=60):
    """是否接近关键 S/R 位置"""
    df = df.copy()
    # 1. 接近 200 日均线 (±2%)
    if "sma200" not in df.columns:
        df["sma200"] = df["close"].rolling(200).mean()
    df["near_sma200"] = (df["close"] - df["sma200"]).abs() / df["sma200"] < 0.02
    df["above_sma200"] = df["close"] > df["sma200"]
    # 2. 接近 60 日 swing high (±2%) / swing low (±2%)
    swing_high = df["high"].rolling(lookback).max()
    swing_low = df["low"].rolling(lookback).min()
    df["near_swing_high"] = (swing_high - df["close"]) / swing_high < 0.02
    df["near_swing_low"] = (df["close"] - swing_low) / swing_low < 0.02
    # 3. 接近 Pivot 计算的 R1/S1 (用前一日 H/L/C)
    prev_h = df["high"].shift(1)
    prev_l = df["low"].shift(1)
    prev_c = df["close"].shift(1)
    pivot = (prev_h + prev_l + prev_c) / 3
    r1 = 2 * pivot - prev_l
    s1 = 2 * pivot - prev_h
    df["near_r1"] = (r1 - df["close"]).abs() / df["close"] < 0.005
    df["near_s1"] = (df["close"] - s1).abs() / df["close"] < 0.005
    # 4. 接近 Bollinger 上下轨 (BB %B 接近 0 或 1)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["near_bb_upper"] = (bb_upper - df["close"]) / df["close"] < 0.01
    df["near_bb_lower"] = (df["close"] - bb_lower) / df["close"] < 0.01
    return df


# ============== Candle patterns ==============
def add_candle_patterns(df):
    """5 类K线形态识别"""
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    avg_body_20 = body.rolling(20).mean()

    # 1. Doji (实体 < 总长 10%)
    df["pat_doji"] = body / rng.replace(0, np.nan) < 0.1
    # 2. Hammer / Hanging Man (实体小, 下影长)
    df["pat_hammer"] = (
        (body / rng.replace(0, np.nan) < 0.3) &
        (lower > 2 * body) &
        (upper < body)
    )
    # 3. Inverted Hammer / Shooting Star (实体小, 上影长)
    df["pat_inverted_hammer"] = (
        (body / rng.replace(0, np.nan) < 0.3) &
        (upper > 2 * body) &
        (lower < body)
    )
    # 4. Bullish/Bearish Engulfing
    prev_body_high = df[["open", "close"]].shift(1).max(axis=1)
    prev_body_low = df[["open", "close"]].shift(1).min(axis=1)
    cur_body_high = df[["open", "close"]].max(axis=1)
    cur_body_low = df[["open", "close"]].min(axis=1)
    is_bull_today = df["close"] > df["open"]
    is_bear_yest = df["close"].shift(1) < df["open"].shift(1)
    df["pat_bull_engulf"] = (
        (cur_body_high > prev_body_high) & (cur_body_low < prev_body_low) &
        is_bull_today & is_bear_yest
    )
    is_bear_today = df["close"] < df["open"]
    is_bull_yest = df["close"].shift(1) > df["open"].shift(1)
    df["pat_bear_engulf"] = (
        (cur_body_high > prev_body_high) & (cur_body_low < prev_body_low) &
        is_bear_today & is_bull_yest
    )
    # 5. Three White Soldiers / Three Black Crows
    bull_3 = ((df["close"] > df["open"]) &
              (df["close"].shift(1) > df["open"].shift(1)) &
              (df["close"].shift(2) > df["open"].shift(2)) &
              (df["close"] > df["close"].shift(1)) &
              (df["close"].shift(1) > df["close"].shift(2)))
    bear_3 = ((df["close"] < df["open"]) &
              (df["close"].shift(1) < df["open"].shift(1)) &
              (df["close"].shift(2) < df["open"].shift(2)) &
              (df["close"] < df["close"].shift(1)) &
              (df["close"].shift(1) < df["close"].shift(2)))
    df["pat_three_white"] = bull_3
    df["pat_three_black"] = bear_3

    return df


# ============== Master ==============
def compute_all_signals(df):
    df = add_macd_cross(df)
    df = add_ma_cross(df)
    df = add_kdj(df)
    df = add_dmi(df)
    df = add_psar(df)
    df = add_sr_proximity(df)
    df = add_candle_patterns(df)
    return df
