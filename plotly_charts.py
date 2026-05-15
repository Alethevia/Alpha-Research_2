"""
Plotly 交互式图表模块
Plotly Interactive Charts
================================
负责构建 K 线主图 + 副图 (MACD / RSI / 成交量 / ADX / TTM 动量)。

设计原则:
  - 一切 label 用英文 (HuggingFace Spaces 默认 Docker 无中文字体)
  - 所有 overlay 由调用方 checkbox 控制 (overlays dict)
  - 不修改输入 df, 只读字段
  - 缺列时静默跳过该 overlay/标记, 不抛错
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Markers shown on the candle subplot. Each entry is a column name in df
# plus styling. Only columns that exist will be plotted.
SIGNAL_MARKERS = [
    # column,                  legend label,            symbol,         color,        anchor
    ("macd_gc",                "MACD GC",               "triangle-up",  "#16a34a",   "below"),
    ("macd_dc",                "MACD DC",               "triangle-down","#dc2626",   "above"),
    ("ma_long_gc",             "Golden Cross 50/200",   "star",         "#15803d",   "below"),
    ("ma_long_dc",             "Death Cross 50/200",    "star",         "#991b1b",   "above"),
    ("pat_double_bottom",      "Double Bottom",         "diamond",      "#2563eb",   "below"),
    ("pat_double_top",         "Double Top",            "x",            "#7f1d1d",   "above"),
    ("pat_inverse_head_shoulders", "Inv H&S",           "diamond-open", "#1d4ed8",   "below"),
    ("pat_head_shoulders",     "Head & Shoulders",      "x-open",       "#7f1d1d",   "above"),
    ("supertrend_flip_up",     "Supertrend Flip ↑",    "triangle-up-open","#059669", "below"),
    ("supertrend_flip_dn",     "Supertrend Flip ↓",    "triangle-down-open","#b91c1c","above"),
    ("squeeze_fired_up",       "TTM Fired ↑",           "circle",       "#0891b2",   "below"),
    ("squeeze_fired_dn",       "TTM Fired ↓",           "circle",       "#9333ea",   "above"),
    ("rsi_bullish_div",        "RSI Bull Div",          "hexagon",      "#16a34a",   "below"),
    ("rsi_bearish_div",        "RSI Bear Div",          "hexagon",      "#dc2626",   "above"),
    ("cd_bullish_div",         "MACD Bull Div",         "hexagon2",     "#16a34a",   "below"),
    ("cd_bearish_div",         "MACD Bear Div",         "hexagon2",     "#dc2626",   "above"),
]


DEFAULT_OVERLAYS = {
    "sma50": True,
    "sma200": True,
    "ema20": False,
    "bollinger": False,
    "keltner": False,
    "supertrend": False,
    "ichimoku": False,
    "vwap": False,
    "ttm_squeeze": False,
}


SUBPLOT_OPTIONS = ["volume", "macd", "rsi", "adx", "ttm_momentum"]


def build_main_chart(
    df: pd.DataFrame,
    symbol: str,
    overlays: dict | None = None,
    subplots: list[str] | None = None,
    show_signal_markers: bool = True,
    show_volume_overlay_colors: bool = True,
    height: int = 900,
) -> go.Figure:
    """
    构建一个 plotly Figure: 主 K 线图 + 用户选择的副图。

    Parameters
    ----------
    df : DataFrame
        必须含 timestamp/open/high/low/close/volume; 可选指标列。
    symbol : str
        Title 中显示。
    overlays : dict
        Map of overlay name -> bool. 缺省为 DEFAULT_OVERLAYS。
    subplots : list[str]
        要显示哪些副图 (按顺序), 元素来自 SUBPLOT_OPTIONS。
        缺省为 ["volume", "macd", "rsi"]。
    show_signal_markers : bool
        是否在 K 线主图叠加金叉/双底等信号标记。
    height : int
        图表总高度 (px)。
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data", height=300)
        return fig

    overlays = {**DEFAULT_OVERLAYS, **(overlays or {})}
    subplots = subplots if subplots is not None else ["volume", "macd", "rsi"]
    subplots = [s for s in subplots if s in SUBPLOT_OPTIONS]

    n_sub = len(subplots)
    n_rows = 1 + n_sub
    # main chart gets larger share; sub charts equal
    row_heights = [0.55] + [0.45 / n_sub] * n_sub if n_sub > 0 else [1.0]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
    )

    x = df["timestamp"]

    # ===== Main: candlestick =====
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            increasing_fillcolor="#16a34a",
            decreasing_fillcolor="#dc2626",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ===== Overlays on main chart =====
    if overlays.get("sma50") and "sma50" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["sma50"], mode="lines", name="SMA50",
            line=dict(color="#f59e0b", width=1),
        ), row=1, col=1)

    if overlays.get("sma200") and "sma200" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["sma200"], mode="lines", name="SMA200",
            line=dict(color="#9333ea", width=1.2),
        ), row=1, col=1)

    if overlays.get("ema20") and "close" in df.columns:
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=x, y=ema20, mode="lines", name="EMA20",
            line=dict(color="#0ea5e9", width=1),
        ), row=1, col=1)

    if overlays.get("bollinger") and {"bb_upper", "bb_lower", "bb_mid"}.issubset(df.columns):
        fig.add_trace(go.Scatter(
            x=x, y=df["bb_upper"], mode="lines", name="BB Upper",
            line=dict(color="#94a3b8", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["bb_lower"], mode="lines", name="BB Lower",
            line=dict(color="#94a3b8", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["bb_mid"], mode="lines", name="BB Mid",
            line=dict(color="#64748b", width=0.8),
        ), row=1, col=1)

    if overlays.get("keltner") and {"kc_upper", "kc_lower", "kc_middle"}.issubset(df.columns):
        fig.add_trace(go.Scatter(
            x=x, y=df["kc_upper"], mode="lines", name="KC Upper",
            line=dict(color="#0891b2", width=1, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["kc_lower"], mode="lines", name="KC Lower",
            line=dict(color="#0891b2", width=1, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["kc_middle"], mode="lines", name="KC Mid",
            line=dict(color="#0e7490", width=0.8),
        ), row=1, col=1)

    if overlays.get("supertrend") and "supertrend" in df.columns:
        # 用方向上色: 多头绿色, 空头红色
        st_vals = df["supertrend"].values
        bull = df.get("supertrend_bullish")
        if bull is not None:
            bull_vals = bull.astype(bool).values
            st_bull = np.where(bull_vals, st_vals, np.nan)
            st_bear = np.where(~bull_vals, st_vals, np.nan)
            fig.add_trace(go.Scatter(
                x=x, y=st_bull, mode="lines", name="Supertrend ↑",
                line=dict(color="#16a34a", width=1.5),
                connectgaps=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=st_bear, mode="lines", name="Supertrend ↓",
                line=dict(color="#dc2626", width=1.5),
                connectgaps=False,
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=x, y=st_vals, mode="lines", name="Supertrend",
                line=dict(color="#475569", width=1.2),
            ), row=1, col=1)

    if overlays.get("ichimoku") and {"senkou_a", "senkou_b"}.issubset(df.columns):
        # 先画 Senkou A 不显示, 再画 Senkou B 用 fill 形成云
        fig.add_trace(go.Scatter(
            x=x, y=df["senkou_a"], mode="lines",
            line=dict(color="rgba(34,197,94,0.4)", width=0.8),
            name="Senkou A", showlegend=True,
        ), row=1, col=1)
        # 云的颜色根据 A vs B 判断: A>B 绿, A<B 红
        # 简化处理: 一律半透明绿
        fig.add_trace(go.Scatter(
            x=x, y=df["senkou_b"], mode="lines",
            line=dict(color="rgba(239,68,68,0.4)", width=0.8),
            name="Senkou B", fill="tonexty",
            fillcolor="rgba(34,197,94,0.10)",
        ), row=1, col=1)
        if "tenkan_sen" in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df["tenkan_sen"], mode="lines",
                line=dict(color="#2563eb", width=0.9),
                name="Tenkan",
            ), row=1, col=1)
        if "kijun_sen" in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df["kijun_sen"], mode="lines",
                line=dict(color="#b45309", width=0.9),
                name="Kijun",
            ), row=1, col=1)

    if overlays.get("vwap") and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["vwap"], mode="lines", name="VWAP(20)",
            line=dict(color="#a855f7", width=1, dash="dashdot"),
        ), row=1, col=1)

    # ===== Signal markers =====
    if show_signal_markers:
        # offset: dynamic vertical offset based on price range
        price_range = float((df["high"].max() - df["low"].min()))
        offset = max(price_range * 0.015, 0.5)
        for col, label, marker_symbol, color, anchor in SIGNAL_MARKERS:
            if col not in df.columns:
                continue
            mask = df[col].fillna(False).astype(bool)
            if not mask.any():
                continue
            sub = df[mask]
            if anchor == "below":
                y_vals = sub["low"] - offset
            else:
                y_vals = sub["high"] + offset
            fig.add_trace(go.Scatter(
                x=sub["timestamp"], y=y_vals,
                mode="markers",
                name=label,
                marker=dict(symbol=marker_symbol, color=color, size=10,
                            line=dict(color="#1f2937", width=0.5)),
                hovertemplate=(
                    f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>Price: %{{customdata:.2f}}<extra></extra>"
                ),
                customdata=sub["close"].values,
                legendgroup="signals",
            ), row=1, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)

    # ===== Subplots =====
    for i, sub_name in enumerate(subplots, start=2):
        if sub_name == "volume":
            _add_volume_subplot(fig, df, row=i, color_by_close=show_volume_overlay_colors)
        elif sub_name == "macd":
            _add_macd_subplot(fig, df, row=i)
        elif sub_name == "rsi":
            _add_rsi_subplot(fig, df, row=i)
        elif sub_name == "adx":
            _add_adx_subplot(fig, df, row=i)
        elif sub_name == "ttm_momentum":
            _add_ttm_momentum_subplot(fig, df, row=i)

    # ===== Layout =====
    fig.update_layout(
        title=dict(text=f"{symbol}", x=0.0, xanchor="left", font=dict(size=16)),
        height=height,
        margin=dict(l=40, r=20, t=50, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=10),
        ),
        hovermode="x unified",
        dragmode="zoom",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#94a3b8", spikethickness=1,
        rangebreaks=[dict(bounds=["sat", "mon"])],  # hide weekends for daily charts
    )
    for r in range(1, n_rows + 1):
        fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", row=r, col=1)
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", row=r, col=1)

    return fig


def _add_volume_subplot(fig, df, row, color_by_close=True):
    if "volume" not in df.columns:
        return
    if color_by_close and "close" in df.columns and "open" in df.columns:
        colors = np.where(df["close"] >= df["open"], "#16a34a", "#dc2626")
    else:
        colors = "#64748b"
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"],
        marker_color=colors, opacity=0.6,
        name="Volume", showlegend=False,
    ), row=row, col=1)
    fig.update_yaxes(title_text="Volume", row=row, col=1)


def _add_macd_subplot(fig, df, row):
    if "macd_hist" not in df.columns:
        return
    hist = df["macd_hist"]
    colors = np.where(hist > 0, "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=hist,
        marker_color=colors, opacity=0.55,
        name="MACD Hist", showlegend=False,
    ), row=row, col=1)
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["macd"], mode="lines",
            line=dict(color="#1d4ed8", width=1),
            name="MACD", showlegend=False,
        ), row=row, col=1)
    if "macd_sig" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["macd_sig"], mode="lines",
            line=dict(color="#ea580c", width=1),
            name="Signal", showlegend=False,
        ), row=row, col=1)
    fig.add_hline(y=0, line=dict(color="#475569", width=0.5), row=row, col=1)
    fig.update_yaxes(title_text="MACD", row=row, col=1)


def _add_rsi_subplot(fig, df, row):
    if "rsi" not in df.columns:
        return
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["rsi"], mode="lines",
        line=dict(color="#7c3aed", width=1.2),
        name="RSI(14)", showlegend=False,
    ), row=row, col=1)
    fig.add_hline(y=70, line=dict(color="#dc2626", width=0.6, dash="dot"), row=row, col=1)
    fig.add_hline(y=30, line=dict(color="#16a34a", width=0.6, dash="dot"), row=row, col=1)
    fig.add_hline(y=50, line=dict(color="#94a3b8", width=0.4, dash="dot"), row=row, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)


def _add_adx_subplot(fig, df, row):
    have_any = any(c in df.columns for c in ["adx", "plus_di", "minus_di"])
    if not have_any:
        return
    if "adx" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["adx"], mode="lines",
            line=dict(color="#7c3aed", width=1.2),
            name="ADX", showlegend=False,
        ), row=row, col=1)
    if "plus_di" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["plus_di"], mode="lines",
            line=dict(color="#16a34a", width=0.9),
            name="+DI", showlegend=False,
        ), row=row, col=1)
    if "minus_di" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["minus_di"], mode="lines",
            line=dict(color="#dc2626", width=0.9),
            name="-DI", showlegend=False,
        ), row=row, col=1)
    fig.add_hline(y=25, line=dict(color="#94a3b8", width=0.4, dash="dot"), row=row, col=1)
    fig.update_yaxes(title_text="ADX/DMI", row=row, col=1)


def _add_ttm_momentum_subplot(fig, df, row):
    if "squeeze_momentum" not in df.columns:
        return
    mom = df["squeeze_momentum"]
    # 4-color scheme like John Carter: rising-positive, falling-positive,
    # falling-negative, rising-negative
    prev = mom.shift(1).fillna(0)
    rising = mom >= prev
    pos = mom >= 0
    color = np.where(
        pos & rising, "#16a34a",
        np.where(pos & ~rising, "#86efac",
                 np.where(~pos & ~rising, "#dc2626", "#fca5a5"))
    )
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=mom,
        marker_color=color, opacity=0.85,
        name="TTM Momentum", showlegend=False,
    ), row=row, col=1)
    # Squeeze on/off as small dots on zero line
    if "squeeze_on" in df.columns:
        on_mask = df["squeeze_on"].astype(bool)
        if on_mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[on_mask, "timestamp"], y=[0] * int(on_mask.sum()),
                mode="markers",
                marker=dict(symbol="circle", color="#dc2626", size=4),
                name="Squeeze On", showlegend=False,
            ), row=row, col=1)
        off_mask = (~on_mask) & df.get("squeeze_off", on_mask).astype(bool)
        if off_mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[off_mask, "timestamp"], y=[0] * int(off_mask.sum()),
                mode="markers",
                marker=dict(symbol="circle", color="#16a34a", size=4),
                name="Squeeze Off", showlegend=False,
            ), row=row, col=1)
    fig.add_hline(y=0, line=dict(color="#475569", width=0.5), row=row, col=1)
    fig.update_yaxes(title_text="TTM Mom", row=row, col=1)
