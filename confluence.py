"""
综合共振评分模块
Indicator Confluence / Resonance Scoring
==========================================
核心思想:
  单一指标容易假信号。多个独立指标同时指向同一方向时, 称为"共振"。
  共振强度越高, 技术面的一致性越强。

重要声明:
  共振是描述性统计, **不是预测**。
  在我们 21 年回测中, 多指标组合并未显著提升预测力。
  共振评分仅帮助用户快速了解多个指标的当前方向性一致程度。
  **它不构成任何买卖建议**。

评分方法:
  1. 每个指标维度独立评分: -1 (看跌) / 0 (中性) / +1 (看涨)
  2. 6 个维度: 趋势、动量、超买超卖、成交量、波动率、形态
  3. 加权汇总得出综合评分 (-100 到 +100)
  4. 多时间框架共振 (日/周/月)

局限性 (必须展示给用户):
  - 所有指标都基于历史价格, 本质上是 lagging indicator
  - 高共振不等于高概率盈利
  - 黑天鹅事件 / 基本面变化会让所有技术指标同时失效
  - 回测中高共振时段的胜率提升幅度极小 (< 2%)
"""

import numpy as np
import pandas as pd


# =============================================================================
# 维度定义
# =============================================================================

# 每个维度包含多个指标, 每个指标贡献 +1 (看涨), -1 (看跌), 或 0 (中性)
# 维度内取平均, 然后按权重汇总

DIMENSION_WEIGHTS = {
    "trend": 0.25,        # 趋势类 (均线排列, Supertrend, Ichimoku 云)
    "momentum": 0.25,     # 动量类 (MACD, TRIX, RSI 方向)
    "overbought_oversold": 0.15,  # 超买超卖 (RSI, CCI, WR, KDJ)
    "volume": 0.10,       # 成交量 (OBV, VWAP)
    "volatility": 0.10,   # 波动率 (ATR, Bollinger)
    "pattern": 0.15,      # 形态 (K线形态, 图表形态, NXCD 背离)
}


def compute_dimension_scores(row, df_full=None):
    """
    给定最新一行数据, 计算 6 个维度的得分。

    Parameters:
        row: pd.Series, df_full.iloc[-1]
        df_full: 完整 DataFrame (用于背离等需要历史的计算)

    Returns:
        dict: {dimension_name: {"score": float, "signals": list, "detail": str}}
    """
    scores = {}

    # === 1. 趋势维度 ===
    trend_signals = []

    # 均线排列
    if row.get("above_sma200"):
        trend_signals.append(("价格在 200 日均线上方", +1))
    else:
        trend_signals.append(("价格在 200 日均线下方", -1))

    # NX 多空通道
    if row.get("nx_bull_zone"):
        trend_signals.append(("NX 多头排列 (EMA5>20>60)", +1))
    elif row.get("nx_bear_zone"):
        trend_signals.append(("NX 空头排列 (EMA5<20<60)", -1))
    else:
        trend_signals.append(("NX 中性 (均线交织)", 0))

    # Supertrend
    if row.get("supertrend_bullish"):
        trend_signals.append(("Supertrend 多头", +1))
    elif row.get("supertrend_bearish"):
        trend_signals.append(("Supertrend 空头", -1))

    # Ichimoku 云
    if row.get("ichimoku_above_cloud"):
        trend_signals.append(("价格在一目云上方", +1))
    elif row.get("ichimoku_below_cloud"):
        trend_signals.append(("价格在一目云下方", -1))
    else:
        trend_signals.append(("价格在一目云内 (盘整区)", 0))

    # PSAR
    if row.get("psar_long"):
        trend_signals.append(("PSAR 多头", +1))
    else:
        trend_signals.append(("PSAR 空头", -1))

    # Heikin-Ashi 强势趋势
    if row.get("ha_strong_bull"):
        trend_signals.append(("HA 强势上升 (3+ 阳无下影)", +1))
    elif row.get("ha_strong_bear"):
        trend_signals.append(("HA 强势下跌 (3+ 阴无上影)", -1))

    scores["trend"] = _summarize_signals("趋势", trend_signals)

    # === 2. 动量维度 ===
    momentum_signals = []

    # MACD
    macd_state = row.get("macd_state")
    if macd_state == 1:
        momentum_signals.append(("MACD 柱状在零轴上方", +1))
    elif macd_state == -1:
        momentum_signals.append(("MACD 柱状在零轴下方", -1))
    else:
        # 检查 macd_hist
        mh = row.get("macd_hist")
        if pd.notna(mh):
            if mh > 0:
                momentum_signals.append(("MACD 柱状 > 0", +1))
            else:
                momentum_signals.append(("MACD 柱状 < 0", -1))

    # RSI 方向
    rsi = row.get("rsi")
    if pd.notna(rsi):
        if rsi > 60:
            momentum_signals.append(("RSI > 60 (偏强)", +1))
        elif rsi < 40:
            momentum_signals.append(("RSI < 40 (偏弱)", -1))
        else:
            momentum_signals.append(("RSI 40-60 (中性)", 0))

    # TRIX
    if row.get("trix_positive"):
        momentum_signals.append(("TRIX > 0 (上涨动量)", +1))
    elif row.get("trix_negative"):
        momentum_signals.append(("TRIX < 0 (下跌动量)", -1))

    # DMI
    adx = row.get("adx")
    plus_di = row.get("plus_di")
    minus_di = row.get("minus_di")
    if pd.notna(adx) and pd.notna(plus_di) and pd.notna(minus_di):
        if adx > 25:
            if plus_di > minus_di:
                momentum_signals.append(("ADX>25 且 +DI > -DI (强势上涨)", +1))
            else:
                momentum_signals.append(("ADX>25 且 -DI > +DI (强势下跌)", -1))

    scores["momentum"] = _summarize_signals("动量", momentum_signals)

    # === 3. 超买超卖维度 ===
    ob_os_signals = []

    if pd.notna(rsi):
        if rsi > 70:
            ob_os_signals.append(("RSI > 70 (超买)", -1))
        elif rsi < 30:
            ob_os_signals.append(("RSI < 30 (超卖)", +1))
        else:
            ob_os_signals.append(("RSI 中性区", 0))

    cci = row.get("cci")
    if pd.notna(cci):
        if cci > 100:
            ob_os_signals.append(("CCI > 100 (超买)", -1))
        elif cci < -100:
            ob_os_signals.append(("CCI < -100 (超卖)", +1))
        else:
            ob_os_signals.append(("CCI 中性区", 0))

    wr = row.get("williams_r")
    if pd.notna(wr):
        if wr > -20:
            ob_os_signals.append(("Williams %R > -20 (超买)", -1))
        elif wr < -80:
            ob_os_signals.append(("Williams %R < -80 (超卖)", +1))
        else:
            ob_os_signals.append(("Williams %R 中性区", 0))

    kdj_k = row.get("kdj_k")
    if pd.notna(kdj_k):
        if kdj_k > 80:
            ob_os_signals.append(("KDJ K > 80 (超买)", -1))
        elif kdj_k < 20:
            ob_os_signals.append(("KDJ K < 20 (超卖)", +1))
        else:
            ob_os_signals.append(("KDJ K 中性区", 0))

    scores["overbought_oversold"] = _summarize_signals("超买超卖", ob_os_signals)

    # === 4. 成交量维度 ===
    volume_signals = []

    if row.get("obv_trend_up"):
        volume_signals.append(("OBV 趋势向上 (量价配合)", +1))
    elif row.get("obv_trend_dn"):
        volume_signals.append(("OBV 趋势向下 (量价背离)", -1))

    if row.get("above_vwap"):
        volume_signals.append(("价格 > VWAP (机构偏多)", +1))
    elif row.get("below_vwap"):
        volume_signals.append(("价格 < VWAP (机构偏空)", -1))

    # CMF (Chaikin Money Flow)
    cmf = row.get("cmf")
    if pd.notna(cmf):
        if cmf > 0.25:
            volume_signals.append(("CMF > 0.25 强烈买盘", +1))
        elif cmf < -0.25:
            volume_signals.append(("CMF < -0.25 强烈卖盘", -1))
        elif cmf > 0:
            volume_signals.append(("CMF > 0 资金流入", +1))
        elif cmf < 0:
            volume_signals.append(("CMF < 0 资金流出", -1))
        else:
            volume_signals.append(("CMF ≈ 0 (中性)", 0))

    scores["volume"] = _summarize_signals("成交量", volume_signals)

    # === 5. 波动率维度 ===
    volatility_signals = []

    if row.get("atr_expanding"):
        volatility_signals.append(("ATR 连续扩大 (波动加剧)", 0))  # 中性: 波动本身无方向
    elif row.get("atr_contracting"):
        volatility_signals.append(("ATR 连续收缩 (可能将突破)", 0))

    if row.get("near_bb_upper"):
        volatility_signals.append(("接近布林上轨", -1))
    elif row.get("near_bb_lower"):
        volatility_signals.append(("接近布林下轨", +1))

    if row.get("atr_high_volatility"):
        volatility_signals.append(("ATR 异常高 (>2x 均值)", 0))
    elif row.get("atr_low_volatility"):
        volatility_signals.append(("ATR 异常低 (<0.5x 均值, 可能酝酿大波动)", 0))

    # TTM Squeeze 释放: 方向不中性 (与动量同向)
    if row.get("squeeze_fired_up"):
        volatility_signals.append(("TTM 挤压向上释放", +1))
    elif row.get("squeeze_fired_dn"):
        volatility_signals.append(("TTM 挤压向下释放", -1))
    elif row.get("squeeze_on"):
        volatility_signals.append(("TTM 挤压中 (低波动, 等待突破)", 0))

    # Keltner 突破
    if row.get("kc_break_up"):
        volatility_signals.append(("Keltner 上轨突破", +1))
    elif row.get("kc_break_dn"):
        volatility_signals.append(("Keltner 下轨突破", -1))

    scores["volatility"] = _summarize_signals("波动率", volatility_signals)

    # === 6. 形态维度 ===
    pattern_signals = []

    # 看涨形态
    bullish_patterns = [
        ("pat_double_bottom", "双底"),
        ("pat_inverse_head_shoulders", "倒头肩"),
        ("pat_bull_engulf", "看涨吞没"),
        ("pat_hammer", "锤子线"),
        ("pat_morning_star", "早晨之星"),
        ("pat_bull_flag", "多头旗形"),
        ("pat_falling_wedge", "下降楔形"),
        ("pat_cup_handle", "杯柄形"),
        ("pat_rounding_bottom", "圆底"),
        ("pat_breakout_up", "向上突破"),
    ]
    for key, name in bullish_patterns:
        if row.get(key):
            pattern_signals.append((f"✓ {name}", +1))

    # 看跌形态
    bearish_patterns = [
        ("pat_double_top", "双顶"),
        ("pat_head_shoulders", "头肩顶"),
        ("pat_bear_engulf", "看跌吞没"),
        ("pat_evening_star", "黄昏之星"),
        ("pat_bear_flag", "空头旗形"),
        ("pat_rising_wedge", "上升楔形"),
        ("pat_breakout_down", "向下突破"),
    ]
    for key, name in bearish_patterns:
        if row.get(key):
            pattern_signals.append((f"✓ {name}", -1))

    # NXCD 背离
    if row.get("cd_bullish_div"):
        pattern_signals.append(("MACD DIF 底背离 (CD)", +1))
    if row.get("cd_bearish_div"):
        pattern_signals.append(("MACD DIF 顶背离 (CD)", -1))
    if row.get("rsi_bullish_div"):
        pattern_signals.append(("RSI 底背离", +1))
    if row.get("rsi_bearish_div"):
        pattern_signals.append(("RSI 顶背离", -1))

    if not pattern_signals:
        pattern_signals.append(("无明显形态", 0))

    scores["pattern"] = _summarize_signals("形态", pattern_signals)

    return scores


def compute_overall_score(dimension_scores):
    """
    加权汇总各维度得分, 得出综合评分 (-100 到 +100)。

    Returns:
        dict: {
            "score": float (-100 ~ +100),
            "label": str (描述性标签),
            "dimensions": dict (各维度详情),
            "disclaimer": str (必须展示的免责声明),
        }
    """
    weighted_sum = 0
    for dim_name, weight in DIMENSION_WEIGHTS.items():
        if dim_name in dimension_scores:
            dim = dimension_scores[dim_name]
            weighted_sum += dim["score"] * weight

    # 归一化到 -100 ~ +100
    overall = max(-100, min(100, weighted_sum * 100))

    # 描述性标签 (注意: 不是买卖建议)
    if overall >= 60:
        label = "强多头共振"
        detail = "多数指标指向上涨方向, 技术面一致性高"
    elif overall >= 30:
        label = "偏多头"
        detail = "多数指标偏向上涨, 但有部分中性或看跌"
    elif overall >= -30:
        label = "中性 / 混合"
        detail = "指标方向不一致, 无明显技术面共识"
    elif overall >= -60:
        label = "偏空头"
        detail = "多数指标偏向下跌, 但有部分中性或看涨"
    else:
        label = "强空头共振"
        detail = "多数指标指向下跌方向, 技术面一致性高"

    return {
        "score": round(overall, 1),
        "label": label,
        "detail": detail,
        "dimensions": dimension_scores,
        "disclaimer": (
            "⚠️ 共振评分是描述性统计, 不是预测, 不构成买卖建议。\n"
            "高共振不等于高概率盈利。在 21 年回测中, 高共振时段的胜率提升极小 (<2%)。\n"
            "技术指标都基于历史价格 (滞后指标), 无法预判黑天鹅事件或基本面变化。\n"
            "所有交易决定为使用者自身责任。"
        ),
    }


def compute_confluence_summary(df_full):
    """
    一次性计算当前最新 bar 的综合共振评分。

    Parameters:
        df_full: 完整 DataFrame (已计算所有指标)

    Returns:
        dict: compute_overall_score 的输出
    """
    if df_full.empty or len(df_full) < 2:
        return {
            "score": 0,
            "label": "数据不足",
            "detail": "无法计算共振评分",
            "dimensions": {},
            "disclaimer": "",
        }

    last = df_full.iloc[-1]
    dimension_scores = compute_dimension_scores(last, df_full)
    return compute_overall_score(dimension_scores)


def _row_confluence_score(row, df_full=None) -> float:
    """轻量版: 只返回 -100~+100 的综合评分, 用于历史回放/窗口稳定性评估."""
    weighted_sum = 0.0
    dim_scores = compute_dimension_scores(row, df_full)
    for dim_name, weight in DIMENSION_WEIGHTS.items():
        if dim_name in dim_scores:
            weighted_sum += dim_scores[dim_name]["score"] * weight
    return max(-100, min(100, weighted_sum * 100))


def find_optimal_window(
    df_full,
    candidate_windows=(30, 60, 90, 180, 365),
    eval_bars=None,  # 已弃用, 保留入参向后兼容
):
    """
    在多个候选显示窗口上测量共振评分稳定性, 返回最稳定的窗口。

    评估方法 (在每个候选窗口的 *全部* bar 上计算):
      - std         : 共振评分标准差 (越小 = 越稳)
      - sign_flips  : 评分穿越 0 的次数 (越少 = 多空切换越少, 信号越一致)
      - flip_rate   : sign_flips / n_bars (按窗口长度归一化, 公平比较)
      - mean, last, range : 描述性辅助字段

    选择规则 (排序依据):
      1) flip_rate 升序 (按时间归一化的方向稳定性, 优先级最高)
      2) std       升序 (波动小的更直观)
      3) -window   升序 (打平时偏好 *较长* 窗口, 信息量更多)

    Parameters
    ----------
    df_full : DataFrame, 已计算全部指标 (含 timestamp 升序)
    candidate_windows : iterable[int], 候选窗口天数 (日历日)
    eval_bars : 已弃用, 忽略

    Returns
    -------
    dict:
      best_window, best_reason, candidates (list of per-window stats)
    """
    import pandas as pd

    if df_full is None or df_full.empty or len(df_full) < 30:
        return {
            "best_window": max(candidate_windows),
            "best_reason": "数据不足, 默认最大窗口",
            "candidates": [],
        }
    if "timestamp" not in df_full.columns:
        return {
            "best_window": max(candidate_windows),
            "best_reason": "缺少 timestamp 列",
            "candidates": [],
        }

    last_ts = df_full["timestamp"].iloc[-1]
    max_window = max(candidate_windows)
    earliest_needed = last_ts - pd.Timedelta(days=max_window)

    # 只对最大候选窗口内的 bar 计算 confluence (避免 O(n_total) 开销)
    span_mask = df_full["timestamp"] >= earliest_needed
    spanned = df_full[span_mask].reset_index(drop=True)
    if len(spanned) < 5:
        return {
            "best_window": max(candidate_windows),
            "best_reason": "窗口内 bar 数太少",
            "candidates": [],
        }
    scores = [_row_confluence_score(spanned.iloc[i], df_full) for i in range(len(spanned))]
    score_series = pd.Series(scores, index=spanned["timestamp"])

    candidates = []
    for w in sorted(candidate_windows):
        window_start = last_ts - pd.Timedelta(days=w)
        ws = score_series[score_series.index >= window_start]
        if len(ws) < 5:
            continue
        sign_flips = int(((ws > 0).astype(int).diff().abs() > 0).sum())
        n_bars = int(len(ws))
        flip_rate = sign_flips / max(1, n_bars - 1)  # 翻转率
        candidates.append({
            "window": w,
            "n_bars": n_bars,
            "std": round(float(ws.std(ddof=0)), 2),
            "mean": round(float(ws.mean()), 2),
            "last": round(float(ws.iloc[-1]), 2),
            "range": round(float(ws.max() - ws.min()), 2),
            "sign_flips": sign_flips,
            "flip_rate": round(flip_rate, 4),
        })

    if not candidates:
        return {
            "best_window": max(candidate_windows),
            "best_reason": "无候选窗口数据足够",
            "candidates": [],
        }

    ranked = sorted(
        candidates,
        key=lambda c: (c["flip_rate"], c["std"], -c["window"]),
    )
    best = ranked[0]
    reason = (
        f"该窗口内方向最稳定 (符号翻转率 {best['flip_rate']*100:.1f}%, "
        f"std={best['std']}, n_bars={best['n_bars']})。"
    )

    return {
        "best_window": best["window"],
        "best_reason": reason,
        "candidates": ranked,
    }


# =============================================================================
# 内部辅助函数
# =============================================================================
def _summarize_signals(dim_name, signals):
    """
    汇总一个维度内所有信号的得分。

    Parameters:
        dim_name: str, 维度名称
        signals: list of (description, score) tuples

    Returns:
        dict: {"score": float (-1~+1), "signals": list, "n_bull": int, "n_bear": int, "n_neutral": int}
    """
    if not signals:
        return {"score": 0, "signals": [], "n_bull": 0, "n_bear": 0, "n_neutral": 0, "detail": "无数据"}

    total = sum(s for _, s in signals)
    count = len(signals)
    avg = total / count  # -1 ~ +1

    n_bull = sum(1 for _, s in signals if s > 0)
    n_bear = sum(1 for _, s in signals if s < 0)
    n_neutral = sum(1 for _, s in signals if s == 0)

    if avg > 0.3:
        detail = f"偏多头 ({n_bull} 多 / {n_bear} 空 / {n_neutral} 中)"
    elif avg < -0.3:
        detail = f"偏空头 ({n_bull} 多 / {n_bear} 空 / {n_neutral} 中)"
    else:
        detail = f"中性 ({n_bull} 多 / {n_bear} 空 / {n_neutral} 中)"

    return {
        "score": avg,
        "signals": signals,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "n_neutral": n_neutral,
        "detail": detail,
    }
