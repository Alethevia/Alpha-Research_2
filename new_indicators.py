"""
新增技术指标模块
New Technical Indicators Module
================================
包含:
1. RSI (相对强弱指标)
2. CCI (顺势指标)
3. Williams %R (威廉指标)
4. OBV (能量潮)
5. ATR (真实波幅)
6. VWAP (成交量加权均价)
7. Supertrend (超级趋势)
8. Ichimoku Cloud (一目均衡表)
9. NXCD 组合指标:
   - NX: 多空通道 (Bull-Bear Channel)
   - CD: MACD 背离检测 (MACD Divergence)
10. TRIX (三重指数平滑)
11. Keltner Channels (肯特纳通道)
12. Chaikin Money Flow (蔡金资金流)
13. TTM Squeeze (TTM 挤压, Bollinger 在 Keltner 内)
14. Heikin-Ashi (平均K线)

每个指标都有:
- 完整计算逻辑 (无占位符)
- 信号触发条件
- 已知局限性说明 (注释中)
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. RSI - Relative Strength Index (相对强弱指标)
# 原理: 衡量一段时间内涨幅与跌幅的比率
# 局限性: 强趋势中会长期钝化在超买/超卖区; 横盘时频繁假信号
# =============================================================================
def add_rsi(df, period=14):
    """
    计算 RSI 指标 + 生成信号
    信号:
      rsi_oversold: RSI < 30 (超卖)
      rsi_overbought: RSI > 70 (超买)
      rsi_cross_up_30: RSI 从下方穿越 30 (超卖区回升)
      rsi_cross_dn_70: RSI 从上方穿越 70 (超买区回落)
      rsi_bullish_div: RSI 底背离 (价格新低但 RSI 没新低)
      rsi_bearish_div: RSI 顶背离 (价格新高但 RSI 没新高)
    """
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # 信号
    df["rsi_oversold"] = (df["rsi"] < 30).astype(bool)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(bool)

    prev_rsi = df["rsi"].shift(1).fillna(50)
    df["rsi_cross_up_30"] = ((df["rsi"] >= 30) & (prev_rsi < 30)).astype(bool)
    df["rsi_cross_dn_70"] = ((df["rsi"] <= 70) & (prev_rsi > 70)).astype(bool)

    # 背离检测 (用最近 60 根 bar 的局部极值)
    df["rsi_bullish_div"] = _detect_divergence(df, "rsi", direction="bullish", lookback=60)
    df["rsi_bearish_div"] = _detect_divergence(df, "rsi", direction="bearish", lookback=60)

    return df


# =============================================================================
# 2. CCI - Commodity Channel Index (顺势指标)
# 原理: 衡量价格偏离统计均值的程度
# 局限性: 趋势市中频繁超买超卖信号; 参数敏感
# =============================================================================
def add_cci(df, period=20):
    """
    CCI = (TP - SMA(TP)) / (0.015 * MAD(TP))
    信号:
      cci_oversold: CCI < -100
      cci_overbought: CCI > +100
      cci_cross_up_minus100: CCI 穿越 -100 向上
      cci_cross_dn_plus100: CCI 穿越 +100 向下
    """
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad_tp = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

    df["cci"] = (tp - sma_tp) / (0.015 * mad_tp.replace(0, np.nan))
    df["cci"] = df["cci"].fillna(0)

    df["cci_oversold"] = (df["cci"] < -100).astype(bool)
    df["cci_overbought"] = (df["cci"] > 100).astype(bool)

    prev = df["cci"].shift(1).fillna(0)
    df["cci_cross_up_minus100"] = ((df["cci"] >= -100) & (prev < -100)).astype(bool)
    df["cci_cross_dn_plus100"] = ((df["cci"] <= 100) & (prev > 100)).astype(bool)

    return df


# =============================================================================
# 3. Williams %R (威廉指标)
# 原理: 类似 KDJ 但更灵敏, 衡量当前价格在 N 日最高最低范围的位置
# 局限性: 极度灵敏, 假信号多; 强趋势中会钝化
# =============================================================================
def add_williams_r(df, period=14):
    """
    %R = (最高价N - 收盘价) / (最高价N - 最低价N) * -100
    信号:
      wr_oversold: WR < -80
      wr_overbought: WR > -20
      wr_cross_up_minus80: WR 从 -80 以下穿越 -80
      wr_cross_dn_minus20: WR 从 -20 以上穿越 -20
    """
    df = df.copy()
    high_n = df["high"].rolling(period).max()
    low_n = df["low"].rolling(period).min()
    hl_range = high_n - low_n
    df["williams_r"] = np.where(hl_range > 0,
                                 (high_n - df["close"]) / hl_range * -100,
                                 -50)

    df["wr_oversold"] = (df["williams_r"] < -80).astype(bool)
    df["wr_overbought"] = (df["williams_r"] > -20).astype(bool)

    prev = df["williams_r"].shift(1).fillna(-50)
    df["wr_cross_up_minus80"] = ((df["williams_r"] >= -80) & (prev < -80)).astype(bool)
    df["wr_cross_dn_minus20"] = ((df["williams_r"] <= -20) & (prev > -20)).astype(bool)

    return df


# =============================================================================
# 4. OBV - On Balance Volume (能量潮)
# 原理: 用成交量确认价格趋势——涨时加量, 跌时减量
# 局限性: 绝对值无意义, 只看趋势; 低流动性股票无效
# =============================================================================
def add_obv(df):
    """
    OBV: 涨日加量, 跌日减量, 平日不变
    信号:
      obv_bullish_div: 价格新低但 OBV 没新低 (量价背离)
      obv_bearish_div: 价格新高但 OBV 没新高
      obv_trend_up: OBV 20日均线向上
      obv_trend_dn: OBV 20日均线向下
    """
    df = df.copy()
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).fillna(0).cumsum()

    obv_sma = df["obv"].rolling(20).mean()
    obv_sma_prev = obv_sma.shift(1)
    df["obv_trend_up"] = (obv_sma > obv_sma_prev).fillna(False).astype(bool)
    df["obv_trend_dn"] = (obv_sma < obv_sma_prev).fillna(False).astype(bool)

    # OBV 背离
    df["obv_bullish_div"] = _detect_divergence(df, "obv", direction="bullish", lookback=60)
    df["obv_bearish_div"] = _detect_divergence(df, "obv", direction="bearish", lookback=60)

    return df


# =============================================================================
# 5. ATR - Average True Range (真实波幅)
# 原理: 衡量波动率, 不判断方向
# 局限性: 纯波动率指标, 不提供方向信号
# =============================================================================
def add_atr(df, period=14):
    """
    ATR = EMA(TrueRange, N)
    信号:
      atr_high_volatility: ATR > 2 * ATR 50日均值 (异常波动)
      atr_low_volatility: ATR < 0.5 * ATR 50日均值 (极低波动, 可能将突破)
      atr_expanding: ATR 连续 3 天扩大 (波动加剧)
      atr_contracting: ATR 连续 3 天收缩 (波动减弱)
    """
    df = df.copy()
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift(1)).abs()
    l_c = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()

    atr_avg = df["atr"].rolling(50).mean()
    df["atr_high_volatility"] = (df["atr"] > 2 * atr_avg).fillna(False).astype(bool)
    df["atr_low_volatility"] = (df["atr"] < 0.5 * atr_avg).fillna(False).astype(bool)

    atr_diff = df["atr"].diff()
    df["atr_expanding"] = ((atr_diff > 0) & (atr_diff.shift(1) > 0) & (atr_diff.shift(2) > 0)).fillna(False).astype(bool)
    df["atr_contracting"] = ((atr_diff < 0) & (atr_diff.shift(1) < 0) & (atr_diff.shift(2) < 0)).fillna(False).astype(bool)

    return df


# =============================================================================
# 6. VWAP - Volume Weighted Average Price (成交量加权均价)
# 原理: 机构常用, 代表当日"公平价格"
# 局限性: 日内指标, 隔夜重置; 日线上意义有限 → 我们用 rolling VWAP
# =============================================================================
def add_vwap(df, period=20):
    """
    Rolling VWAP = sum(close * volume, N) / sum(volume, N)
    信号:
      above_vwap: 价格在 VWAP 上方 (机构看多区)
      below_vwap: 价格在 VWAP 下方 (机构看空区)
      vwap_cross_up: 价格从下方穿越 VWAP
      vwap_cross_dn: 价格从上方穿越 VWAP
    """
    df = df.copy()
    vol_sum = df["volume"].rolling(period).sum()
    pv_sum = (df["close"] * df["volume"]).rolling(period).sum()
    df["vwap"] = np.where(vol_sum > 0, pv_sum / vol_sum, df["close"])

    df["above_vwap"] = (df["close"] > df["vwap"]).astype(bool)
    df["below_vwap"] = (df["close"] < df["vwap"]).astype(bool)

    prev_above = (df["close"].shift(1) > df["vwap"].shift(1)).fillna(False)
    df["vwap_cross_up"] = (df["above_vwap"] & ~prev_above.astype(bool)).astype(bool)
    df["vwap_cross_dn"] = (df["below_vwap"] & prev_above.astype(bool)).astype(bool)

    return df


# =============================================================================
# 7. Supertrend (超级趋势)
# 原理: 基于 ATR 的趋势跟踪带, 类似 PSAR 但更平滑
# 局限性: 趋势指标, 横盘时反复翻转; 参数 (multiplier) 敏感
# =============================================================================
def add_supertrend(df, period=10, multiplier=3.0):
    """
    Supertrend = ATR-based trailing stop
    信号:
      supertrend_bullish: 价格在 Supertrend 上方 (上升趋势)
      supertrend_bearish: 价格在 Supertrend 下方 (下降趋势)
      supertrend_flip_up: 从空头翻转为多头
      supertrend_flip_dn: 从多头翻转为空头
    """
    df = df.copy()
    n = len(df)

    # ATR
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift(1)).abs()
    l_c = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    hl2 = (df["high"] + df["low"]) / 2
    upper_band = (hl2 + multiplier * atr).values.copy()
    lower_band = (hl2 - multiplier * atr).values.copy()
    close = df["close"].values.copy()

    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=bool)  # True = bullish

    supertrend[0] = upper_band[0]
    direction[0] = close[0] > upper_band[0]

    for i in range(1, n):
        if direction[i-1]:  # 之前是多头
            # lower_band 只能上升不能下降 (trailing stop)
            lower_band[i] = max(lower_band[i], lower_band[i-1]) if close[i-1] > lower_band[i-1] else lower_band[i]
            if close[i] < lower_band[i]:
                direction[i] = False
                supertrend[i] = upper_band[i]
            else:
                direction[i] = True
                supertrend[i] = lower_band[i]
        else:  # 之前是空头
            upper_band[i] = min(upper_band[i], upper_band[i-1]) if close[i-1] < upper_band[i-1] else upper_band[i]
            if close[i] > upper_band[i]:
                direction[i] = True
                supertrend[i] = lower_band[i]
            else:
                direction[i] = False
                supertrend[i] = upper_band[i]

    df["supertrend"] = supertrend
    df["supertrend_bullish"] = direction.astype(bool)
    df["supertrend_bearish"] = (~direction).astype(bool)

    prev_dir = pd.Series(direction).shift(1).fillna(True).astype(bool).values
    df["supertrend_flip_up"] = (direction & ~prev_dir).astype(bool)
    df["supertrend_flip_dn"] = (~direction & prev_dir).astype(bool)

    return df


# =============================================================================
# 8. Ichimoku Cloud (一目均衡表)
# 原理: 日本交易员开发, 用 5 条线构建"云", 判断趋势/支撑/阻力
# 局限性: 参数基于日本旧交易周(9/26/52); 横盘时无效; 算法复杂
# =============================================================================
def add_ichimoku(df, tenkan=9, kijun=26, senkou_b=52, displacement=26):
    """
    Tenkan-sen (转换线): (9日最高+9日最低)/2
    Kijun-sen (基准线): (26日最高+26日最低)/2
    Senkou Span A (先行 A): (转换线+基准线)/2, 前移26日
    Senkou Span B (先行 B): (52日最高+52日最低)/2, 前移26日
    Chikou Span (迟行线): 收盘价后移26日

    信号:
      ichimoku_above_cloud: 价格在云上方 (强势)
      ichimoku_below_cloud: 价格在云下方 (弱势)
      ichimoku_tk_cross_up: 转换线穿越基准线向上 (金叉)
      ichimoku_tk_cross_dn: 转换线穿越基准线向下 (死叉)
      ichimoku_cloud_bullish: 云层颜色看涨 (Senkou A > Senkou B)
    """
    df = df.copy()

    # Tenkan-sen
    h9 = df["high"].rolling(tenkan).max()
    l9 = df["low"].rolling(tenkan).min()
    df["tenkan_sen"] = (h9 + l9) / 2

    # Kijun-sen
    h26 = df["high"].rolling(kijun).max()
    l26 = df["low"].rolling(kijun).min()
    df["kijun_sen"] = (h26 + l26) / 2

    # Senkou Span A (前移 displacement)
    df["senkou_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(displacement)

    # Senkou Span B (前移 displacement)
    h52 = df["high"].rolling(senkou_b).max()
    l52 = df["low"].rolling(senkou_b).min()
    df["senkou_b"] = ((h52 + l52) / 2).shift(displacement)

    # 云的上下界
    cloud_top = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).max(axis=1)
    cloud_bottom = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).min(axis=1)

    df["ichimoku_above_cloud"] = (df["close"] > cloud_top).fillna(False).astype(bool)
    df["ichimoku_below_cloud"] = (df["close"] < cloud_bottom).fillna(False).astype(bool)

    # TK 交叉
    tk_above = df["tenkan_sen"] > df["kijun_sen"]
    tk_above_prev = tk_above.shift(1).fillna(False).astype(bool)
    df["ichimoku_tk_cross_up"] = (tk_above & ~tk_above_prev).astype(bool)
    df["ichimoku_tk_cross_dn"] = (~tk_above & tk_above_prev).astype(bool)

    # 云颜色
    df["ichimoku_cloud_bullish"] = (df["senkou_a"] > df["senkou_b"]).fillna(False).astype(bool)

    return df


# =============================================================================
# 9. NXCD 组合指标 (基于公开原理实现, 非 xxpfly 私有版)
# NX: 多空通道 — 用 EMA 双通道判断牛熊
# CD: MACD 背离 — 检测价格与 MACD DIF 线的顶背离/底背离
# 局限性: NX 横盘时频繁翻转; CD 背离不等于立刻反转, "背离后还有背离"
# =============================================================================
def add_nxcd(df, short_ema=5, mid_ema=20, long_ema=60):
    """
    NX 多空通道:
      nx_bull_zone: 短期 EMA > 中期 EMA > 长期 EMA (完美多头排列)
      nx_bear_zone: 短期 EMA < 中期 EMA < 长期 EMA (完美空头排列)
      nx_channel_width: 通道宽度 (短期EMA - 长期EMA) / 长期EMA * 100
      nx_flip_bull: 从空头区翻转为多头区
      nx_flip_bear: 从多头区翻转为空头区

    CD MACD 背离:
      cd_bullish_div: MACD DIF 底背离 (价格新低但 DIF 不新低)
      cd_bearish_div: MACD DIF 顶背离 (价格新高但 DIF 不新高)
      cd_hist_bullish_div: MACD 柱状 底背离
      cd_hist_bearish_div: MACD 柱状 顶背离
    """
    df = df.copy()

    # --- NX: 多空通道 ---
    df["ema_short_nx"] = df["close"].ewm(span=short_ema, adjust=False).mean()
    df["ema_mid_nx"] = df["close"].ewm(span=mid_ema, adjust=False).mean()
    df["ema_long_nx"] = df["close"].ewm(span=long_ema, adjust=False).mean()

    df["nx_bull_zone"] = (
        (df["ema_short_nx"] > df["ema_mid_nx"]) &
        (df["ema_mid_nx"] > df["ema_long_nx"])
    ).astype(bool)

    df["nx_bear_zone"] = (
        (df["ema_short_nx"] < df["ema_mid_nx"]) &
        (df["ema_mid_nx"] < df["ema_long_nx"])
    ).astype(bool)

    # 通道宽度 (百分比)
    df["nx_channel_width"] = np.where(
        df["ema_long_nx"] > 0,
        (df["ema_short_nx"] - df["ema_long_nx"]) / df["ema_long_nx"] * 100,
        0
    )

    # 翻转
    prev_bull = df["nx_bull_zone"].shift(1).fillna(False).astype(bool)
    prev_bear = df["nx_bear_zone"].shift(1).fillna(False).astype(bool)
    df["nx_flip_bull"] = (df["nx_bull_zone"] & ~prev_bull).astype(bool)
    df["nx_flip_bear"] = (df["nx_bear_zone"] & ~prev_bear).astype(bool)

    # --- CD: MACD 背离 ---
    # 用标准 MACD 参数 (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = 2 * (dif - dea)

    df["macd_dif_nx"] = dif
    df["macd_hist_nx"] = macd_hist

    # DIF 背离
    df["cd_bullish_div"] = _detect_divergence(df, "macd_dif_nx", direction="bullish", lookback=60)
    df["cd_bearish_div"] = _detect_divergence(df, "macd_dif_nx", direction="bearish", lookback=60)

    # 柱状 背离
    df["cd_hist_bullish_div"] = _detect_divergence(df, "macd_hist_nx", direction="bullish", lookback=60)
    df["cd_hist_bearish_div"] = _detect_divergence(df, "macd_hist_nx", direction="bearish", lookback=60)

    return df


# =============================================================================
# 10. TRIX (三重指数平滑)
# 原理: EMA(EMA(EMA(close))) 的变化率, 过滤噪音最强的动量指标
# 局限性: 极度滞后, 短线几乎无用
# =============================================================================
def add_trix(df, period=15, signal=9):
    """
    TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    信号:
      trix_cross_up: TRIX 穿越信号线向上
      trix_cross_dn: TRIX 穿越信号线向下
      trix_positive: TRIX > 0 (上涨动量)
      trix_negative: TRIX < 0 (下跌动量)
    """
    df = df.copy()
    ema1 = df["close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    df["trix"] = 100 * ema3.pct_change()
    df["trix_signal"] = df["trix"].ewm(span=signal, adjust=False).mean()

    df["trix_positive"] = (df["trix"] > 0).fillna(False).astype(bool)
    df["trix_negative"] = (df["trix"] < 0).fillna(False).astype(bool)

    prev_above = (df["trix"].shift(1) > df["trix_signal"].shift(1)).fillna(False).astype(bool)
    curr_above = (df["trix"] > df["trix_signal"]).fillna(False).astype(bool)
    df["trix_cross_up"] = (curr_above & ~prev_above).astype(bool)
    df["trix_cross_dn"] = (~curr_above & prev_above).astype(bool)

    return df


# =============================================================================
# 11. Keltner Channels (肯特纳通道)
# 原理: EMA 中线 ± multiplier * ATR, 类似 Bollinger 但用 ATR 而非标准差
# 局限性: 对 multiplier 敏感; 突破信号在横盘市频繁假信号
# =============================================================================
def add_keltner(df, ema_period=20, atr_period=10, multiplier=2.0):
    """
    Keltner Channels:
      Middle = EMA(close, ema_period)
      Upper  = Middle + multiplier * ATR(atr_period)
      Lower  = Middle - multiplier * ATR(atr_period)

    信号:
      kc_above_upper: 收盘 > 上轨 (强势突破)
      kc_below_lower: 收盘 < 下轨 (弱势突破)
      kc_inside: 收盘在通道内 (正常区间)
      kc_break_up: 当根从内部突破到上方
      kc_break_dn: 当根从内部突破到下方
    """
    df = df.copy()

    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift(1)).abs()
    l_c = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    atr_kc = tr.ewm(span=atr_period, adjust=False).mean()

    df["kc_middle"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    df["kc_upper"] = df["kc_middle"] + multiplier * atr_kc
    df["kc_lower"] = df["kc_middle"] - multiplier * atr_kc

    above = df["close"] > df["kc_upper"]
    below = df["close"] < df["kc_lower"]
    df["kc_above_upper"] = above.fillna(False).astype(bool)
    df["kc_below_lower"] = below.fillna(False).astype(bool)
    df["kc_inside"] = (~df["kc_above_upper"] & ~df["kc_below_lower"]).astype(bool)

    prev_above = above.shift(1).fillna(False).astype(bool)
    prev_below = below.shift(1).fillna(False).astype(bool)
    df["kc_break_up"] = (df["kc_above_upper"] & ~prev_above).astype(bool)
    df["kc_break_dn"] = (df["kc_below_lower"] & ~prev_below).astype(bool)

    return df


# =============================================================================
# 12. Chaikin Money Flow (CMF, 蔡金资金流)
# 原理: 用收盘在 K 线内的位置 + 成交量, 衡量买卖压力
# 局限性: 跳空缺口扭曲读数; 低成交量股票噪音大
# =============================================================================
def add_cmf(df, period=20):
    """
    Money Flow Multiplier = ((C-L) - (H-C)) / (H-L)
    Money Flow Volume     = MFM * Volume
    CMF = sum(MFV, period) / sum(Volume, period)

    信号:
      cmf_positive: CMF > 0 (买盘资金流入)
      cmf_negative: CMF < 0 (卖盘资金流出)
      cmf_strong_bull: CMF > 0.25 (强烈买盘)
      cmf_strong_bear: CMF < -0.25 (强烈卖盘)
      cmf_cross_zero_up: CMF 上穿 0
      cmf_cross_zero_dn: CMF 下穿 0
    """
    df = df.copy()

    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
    mfm = mfm.fillna(0)
    mfv = mfm * df["volume"]

    vol_sum = df["volume"].rolling(period).sum().replace(0, np.nan)
    df["cmf"] = (mfv.rolling(period).sum() / vol_sum).fillna(0)

    df["cmf_positive"] = (df["cmf"] > 0).astype(bool)
    df["cmf_negative"] = (df["cmf"] < 0).astype(bool)
    df["cmf_strong_bull"] = (df["cmf"] > 0.25).astype(bool)
    df["cmf_strong_bear"] = (df["cmf"] < -0.25).astype(bool)

    prev = df["cmf"].shift(1).fillna(0)
    df["cmf_cross_zero_up"] = ((df["cmf"] > 0) & (prev <= 0)).astype(bool)
    df["cmf_cross_zero_dn"] = ((df["cmf"] < 0) & (prev >= 0)).astype(bool)

    return df


# =============================================================================
# 13. TTM Squeeze (TTM 挤压)
# 原理: John Carter 的低波动率检测 — Bollinger Bands 完全在 Keltner 内 = 挤压
#       挤压释放时(BB 重新走出 KC) + 动量方向 = 突破方向
# 局限性: 只检测条件, 不预测方向 (要配合动量); 假突破常见
# 依赖: 需先计算 Bollinger Bands (extended_signals 已计算 bb_upper/bb_lower)
#       和 Keltner Channels (本模块 add_keltner)
# =============================================================================
def add_ttm_squeeze(df, momentum_period=20):
    """
    依赖列: bb_upper, bb_lower, kc_upper, kc_lower

    信号:
      squeeze_on: BB 完全在 KC 内 (压缩, 等待突破)
      squeeze_off: BB 至少一侧在 KC 外 (释放)
      squeeze_fired_up: 上一根 squeeze_on 且当根 squeeze_off, 动量 > 0 (向上释放)
      squeeze_fired_dn: 上一根 squeeze_on 且当根 squeeze_off, 动量 < 0 (向下释放)
      squeeze_momentum: 动量值 (用线性回归斜率近似, 用于柱状)
    """
    df = df.copy()

    needed = ["bb_upper", "bb_lower", "kc_upper", "kc_lower"]
    for col in needed:
        if col not in df.columns:
            # 调用方未计算 BB/KC, 输出全 False, 不抛错
            df["squeeze_on"] = False
            df["squeeze_off"] = False
            df["squeeze_fired_up"] = False
            df["squeeze_fired_dn"] = False
            df["squeeze_momentum"] = 0.0
            return df

    on = (df["bb_upper"] <= df["kc_upper"]) & (df["bb_lower"] >= df["kc_lower"])
    df["squeeze_on"] = on.fillna(False).astype(bool)
    df["squeeze_off"] = (~df["squeeze_on"]).astype(bool)

    # 动量: 收盘相对 (最近 N 根的 (高低中点 + KC 中线)/2 的偏差) 的线性回归值
    # John Carter 原版用线性回归取最后一个值, 这里用 close - midpoint 的 EMA 近似
    n = momentum_period
    high_n = df["high"].rolling(n).max()
    low_n = df["low"].rolling(n).min()
    sma_n = df["close"].rolling(n).mean()
    midpoint = (((high_n + low_n) / 2) + sma_n) / 2
    delta = df["close"] - midpoint

    # 用滚动线性回归的最后一点(斜率 * (n-1) + 截距 = 拟合的当根值)
    def _linreg_last(series):
        if series.isna().any():
            return np.nan
        x = np.arange(len(series), dtype=float)
        y = series.values
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        intercept = y_mean - slope * x_mean
        return slope * x[-1] + intercept

    df["squeeze_momentum"] = delta.rolling(n).apply(_linreg_last, raw=False).fillna(0)

    # Fired: 上一根 squeeze_on, 当根 squeeze_off
    prev_on = df["squeeze_on"].shift(1).fillna(False).astype(bool)
    fired = prev_on & df["squeeze_off"]
    df["squeeze_fired_up"] = (fired & (df["squeeze_momentum"] > 0)).astype(bool)
    df["squeeze_fired_dn"] = (fired & (df["squeeze_momentum"] < 0)).astype(bool)

    return df


# =============================================================================
# 14. Heikin-Ashi (平均K线)
# 原理: 用平滑后的 OHLC 重新画 K 线, 过滤掉短线噪音, 让趋势更直观
# 局限性: 不是真实价格, 不能用于精确入场点; 拐点会滞后
# =============================================================================
def add_heikin_ashi(df):
    """
    HA-Close = (O + H + L + C) / 4
    HA-Open  = (prev HA-Open + prev HA-Close) / 2  (首根 = (O+C)/2)
    HA-High  = max(H, HA-Open, HA-Close)
    HA-Low   = min(L, HA-Open, HA-Close)

    信号:
      ha_bullish: HA 阳线 (HA-Close > HA-Open)
      ha_bearish: HA 阴线 (HA-Close < HA-Open)
      ha_strong_bull: 连续 3 根阳线且无下影线 (强势上升)
      ha_strong_bear: 连续 3 根阴线且无上影线 (强势下跌)
      ha_doji: HA 十字星 (实体极小, 趋势可能反转)
      ha_flip_bull: 前一根阴, 当根阳 (转涨)
      ha_flip_bear: 前一根阳, 当根阴 (转跌)
    """
    df = df.copy()
    n = len(df)

    if n == 0:
        for col in ["ha_open", "ha_high", "ha_low", "ha_close"]:
            df[col] = pd.Series(dtype=float)
        for col in ["ha_bullish", "ha_bearish", "ha_strong_bull", "ha_strong_bear",
                    "ha_doji", "ha_flip_bull", "ha_flip_bear"]:
            df[col] = pd.Series(dtype=bool)
        return df

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.zeros(n)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    df["ha_open"] = ha_open
    df["ha_high"] = ha_high
    df["ha_low"] = ha_low
    df["ha_close"] = ha_close

    bullish = ha_close > ha_open
    bearish = ha_close < ha_open
    df["ha_bullish"] = bullish.astype(bool)
    df["ha_bearish"] = bearish.astype(bool)

    # 强势: 连续 3 根 + 无对应方向影线
    no_lower_wick = ha_low >= np.minimum(ha_open, ha_close) - 1e-9
    no_upper_wick = ha_high <= np.maximum(ha_open, ha_close) + 1e-9

    bull_series = pd.Series(bullish & no_lower_wick)
    bear_series = pd.Series(bearish & no_upper_wick)
    df["ha_strong_bull"] = (
        bull_series & bull_series.shift(1).fillna(False) & bull_series.shift(2).fillna(False)
    ).fillna(False).astype(bool)
    df["ha_strong_bear"] = (
        bear_series & bear_series.shift(1).fillna(False) & bear_series.shift(2).fillna(False)
    ).fillna(False).astype(bool)

    # Doji: 实体小于平均实体的 10%
    body = np.abs(ha_close - ha_open)
    body_avg = pd.Series(body).rolling(20).mean().bfill().values
    df["ha_doji"] = (body < 0.1 * np.where(body_avg > 0, body_avg, 1e-9)).astype(bool)

    prev_bull = pd.Series(bullish).shift(1).fillna(False).astype(bool).values
    prev_bear = pd.Series(bearish).shift(1).fillna(False).astype(bool).values
    df["ha_flip_bull"] = (bullish & prev_bear).astype(bool)
    df["ha_flip_bear"] = (bearish & prev_bull).astype(bool)

    return df


# =============================================================================
# 背离检测通用函数 (供 RSI, OBV, MACD 等多个指标调用)
# =============================================================================
def _detect_divergence(df, indicator_col, direction="bullish", lookback=60, min_swing=5):
    """
    检测价格与指标之间的背离。

    底背离 (bullish): 价格创新低, 但指标没创新低 → 空头动能衰竭
    顶背离 (bearish): 价格创新高, 但指标没创新高 → 多头动能衰竭

    方法: 找最近两个局部极值点, 比较价格走势和指标走势方向是否一致。

    Parameters:
      df: DataFrame, 必须有 'close' 列和 indicator_col 列
      indicator_col: 指标列名
      direction: 'bullish' (底背离) 或 'bearish' (顶背离)
      lookback: 向前看多少根 bar 寻找极值
      min_swing: 两个极值之间最少相隔几根 bar

    Returns:
      pd.Series of bool
    """
    n = len(df)
    result = np.zeros(n, dtype=bool)

    if indicator_col not in df.columns:
        return pd.Series(result, index=df.index)

    close = df["close"].values
    indicator = df[indicator_col].values

    for i in range(lookback + min_swing, n):
        window_close = close[i - lookback:i + 1]
        window_ind = indicator[i - lookback:i + 1]

        if np.any(np.isnan(window_ind)):
            continue

        if direction == "bullish":
            # 找最近两个局部低点
            # 当前低点
            curr_low_price = window_close[-1]
            curr_low_ind = window_ind[-1]

            # 之前的低点 (在 lookback 窗口的前半段)
            half = len(window_close) // 2
            prev_low_idx = np.argmin(window_close[:half])
            prev_low_price = window_close[prev_low_idx]
            prev_low_ind = window_ind[prev_low_idx]

            # 底背离: 价格新低, 但指标没新低
            if (curr_low_price < prev_low_price and
                curr_low_ind > prev_low_ind and
                curr_low_price == np.min(window_close[-min_swing:])):
                result[i] = True

        elif direction == "bearish":
            # 找最近两个局部高点
            curr_high_price = window_close[-1]
            curr_high_ind = window_ind[-1]

            half = len(window_close) // 2
            prev_high_idx = np.argmax(window_close[:half])
            prev_high_price = window_close[prev_high_idx]
            prev_high_ind = window_ind[prev_high_idx]

            # 顶背离: 价格新高, 但指标没新高
            if (curr_high_price > prev_high_price and
                curr_high_ind < prev_high_ind and
                curr_high_price == np.max(window_close[-min_swing:])):
                result[i] = True

    return pd.Series(result, index=df.index).astype(bool)


# =============================================================================
# 主入口: 计算所有新指标
# =============================================================================
def compute_new_indicators(df):
    """
    一次性计算所有新指标。
    顺序很重要:
      - add_keltner 必须在 add_ttm_squeeze 之前 (TTM 依赖 KC 列)
      - TTM 还依赖 bb_upper/bb_lower; 这些列由 extended_signals.compute_all_signals 提供,
        因此 compute_new_indicators 应在 compute_all_signals 之后调用 (app.py 已如此)。
    """
    df = add_rsi(df)
    df = add_cci(df)
    df = add_williams_r(df)
    df = add_obv(df)
    df = add_atr(df)
    df = add_vwap(df)
    df = add_supertrend(df)
    df = add_ichimoku(df)
    df = add_nxcd(df)
    df = add_trix(df)
    df = add_keltner(df)
    df = add_cmf(df)
    df = add_ttm_squeeze(df)
    df = add_heikin_ashi(df)
    return df
