"""
Gradio 替代入口 (可选 / 备用)
=============================
若 Streamlit 在 HuggingFace Space 上出现频繁抖动 (重渲染) 或会话崩溃,
可切换到 Gradio。Gradio 是函数式回调 + 单次提交模型, 不会因 widget
state 变化触发整页重跑, 在弱网/云环境更稳。

启动:
  python gradio_app.py
  # 或 在 HuggingFace Space:
  # SDK 改为 gradio, 入口改为 gradio_app.py

功能覆盖 (核心子集):
  - 单股分析: 输入代码 + 时间窗口 → 加载数据
  - Plotly K 线图 (复用 plotly_charts.build_main_chart)
  - 综合共振评分 + 维度详情
  - 当前触发的信号列表 (含 21 年回测频率)
  - AR(5) Conformal 概率预测

不包含 (留给后续按需补全):
  - 多周期共振 tab
  - 股票池扫描 tab
  - Finnhub 面板 (易扩展, 但代码量较大)
  - Chronos AI (gradio.Slider + Button 即可实现, 略)
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import gradio as gr
import yfinance as yf

from extended_signals import compute_all_signals
from advanced_patterns import detect_all_advanced_patterns
from extra_patterns import detect_extra_patterns
from new_indicators import compute_new_indicators
from confluence import compute_confluence_summary, find_optimal_window
from historical_stats import SIGNAL_STATS, BASELINE_10D
from plotly_charts import build_main_chart
from conformal import probability_forecast


CACHE_DIR = Path.home() / ".freq_app_cache_gradio"
CACHE_DIR.mkdir(exist_ok=True)


def _fetch_data(symbol: str, days_back: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back + 250)
    cache = CACHE_DIR / f"{symbol}_1d_{start.date()}_{end.date()}.pkl"
    if cache.exists():
        try:
            return pd.read_pickle(cache)
        except Exception:
            pass
    try:
        d = yf.Ticker(symbol).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=True,
        )
        if d.empty:
            return pd.DataFrame()
        d = d.reset_index()
        d.columns = [c.lower().replace(" ", "_") for c in d.columns]
        if "date" in d.columns:
            d = d.rename(columns={"date": "timestamp"})
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d[["timestamp", "open", "high", "low", "close", "volume"]]
        d.to_pickle(cache)
        return d
    except Exception:
        return pd.DataFrame()


def analyze(symbol: str,
            window_choice: str,
            show_overlays: list,
            show_subplots: list,
            pf_horizon: int,
            pf_confidence: str):
    """
    Gradio 主回调: 输入控件 → 输出 (chart, summary_md, signals_df, prob_md, error)。
    任何异常都被 catch, 通过 error 栏返回, 不让 Gradio session 崩溃。
    """
    try:
        sym = (symbol or "AAPL").strip().upper()
        period_to_days = {
            "自动 (最稳定)": None,
            "1 个月": 30, "2 个月": 60, "3 个月": 90,
            "6 个月": 180, "1 年": 365, "2 年": 730,
        }
        days = period_to_days.get(window_choice, 365) or 365

        d = _fetch_data(sym, days)
        if d.empty or len(d) < 250:
            return None, "❌ 数据不足或加载失败", pd.DataFrame(), "", "需要至少 250 个交易日"

        df_full = compute_all_signals(d.copy())
        df_full = detect_all_advanced_patterns(df_full)
        df_full = detect_extra_patterns(df_full)
        df_full = compute_new_indicators(df_full)

        if window_choice == "自动 (最稳定)":
            wd = find_optimal_window(df_full, candidate_windows=(30, 60, 90, 180, 365))
            days = wd["best_window"]
            auto_note = f" (自动选择: {days} 天 — {wd['best_reason']})"
        else:
            auto_note = ""

        end = datetime.now(timezone.utc)
        display_start = end - timedelta(days=days)
        disp = df_full[df_full["timestamp"] >= pd.Timestamp(display_start)].copy()
        if disp.empty:
            return None, "❌ 选定窗口无数据", pd.DataFrame(), "", ""

        # === Chart ===
        overlays = {
            "sma50": "SMA50" in show_overlays,
            "sma200": "SMA200" in show_overlays,
            "bollinger": "Bollinger" in show_overlays,
            "keltner": "Keltner" in show_overlays,
            "supertrend": "Supertrend" in show_overlays,
            "ichimoku": "Ichimoku" in show_overlays,
            "vwap": "VWAP" in show_overlays,
        }
        subplots_map = {
            "Volume": "volume", "MACD": "macd", "RSI": "rsi",
            "ADX": "adx", "TTM Mom": "ttm_momentum",
        }
        subs = [subplots_map[s] for s in show_subplots if s in subplots_map]

        fig = build_main_chart(
            disp, sym, overlays=overlays, subplots=subs,
            show_signal_markers=True,
            height=500 + 150 * len(subs),
        )

        # === Confluence + summary markdown ===
        conf = compute_confluence_summary(df_full)
        last = df_full.iloc[-1]
        prev = df_full.iloc[-2]
        change_pct = (last["close"] / prev["close"] - 1) * 100

        summary_md = (
            f"## {sym}\n\n"
            f"- **当前收盘**: ${last['close']:.2f} ({change_pct:+.2f}%)\n"
            f"- **窗口**: {days} 天{auto_note}\n"
            f"- **共振评分**: {conf['score']:.0f} / 100 → **{conf['label']}**\n"
            f"- {conf['detail']}\n\n"
            f"⚠️ {conf['disclaimer']}"
        )

        # === Triggered signals ===
        sig_rows = []
        for sig_col, stat in SIGNAL_STATS.items():
            if sig_col in df_full.columns:
                val = last.get(sig_col)
                if isinstance(val, (bool, np.bool_)) and bool(val):
                    sig_rows.append({
                        "信号": stat["label"],
                        "类别": stat["category"],
                        "n": stat["n"],
                        "21年上涨率%": round(stat["pct_up_10d"], 1),
                        "近6年上涨率%": round(stat.get("test_pct_up_10d", stat["pct_up_10d"]), 1),
                        "Alpha%": round(stat["alpha_10d"], 2),
                        "p值": round(stat["p_value"], 3),
                        "近6年仍显著": "✓" if stat.get("oos_significant") else "✗",
                    })
        sig_rows.sort(key=lambda r: (r["近6年仍显著"] != "✓", -abs(r["Alpha%"])))
        sig_df = pd.DataFrame(sig_rows) if sig_rows else pd.DataFrame(
            columns=["信号", "类别", "n", "21年上涨率%", "近6年上涨率%", "Alpha%", "p值", "近6年仍显著"]
        )

        # === Probability forecast ===
        alpha_map = {"80%": 0.20, "90%": 0.10, "95%": 0.05}
        alpha = alpha_map.get(pf_confidence, 0.10)
        pf = probability_forecast(df_full, horizon=int(pf_horizon), alpha=alpha)
        if pf["ok"]:
            prob_md = (
                f"### 概率预测 (AR(5) + Split Conformal)\n\n"
                f"- **{int(pf_horizon)}d 点预测**: {pf['point_pct']:+.2f}%\n"
                f"- **{int((1 - alpha) * 100)}% 置信区间**: "
                f"[{pf['lower_pct']:+.1f}%, {pf['upper_pct']:+.1f}%]\n"
                f"- **P(上涨)**: {pf['p_up'] * 100:.1f}% (基线 {pf['naive_p_up_baseline'] * 100:.1f}%)\n"
                f"- **P(>+5%) / P(<-5%)**: {pf['p_up_5pct'] * 100:.1f}% / {pf['p_dn_5pct'] * 100:.1f}%\n"
                f"- **校准实测覆盖**: {pf['calibration_coverage'] * 100:.1f}% (名义 {(1 - alpha) * 100:.0f}%)\n\n"
                f"⚠️ AR(5) 是低复杂度基线; 不构成投资建议; 预测精度有限."
            )
        else:
            prob_md = f"⚠️ 概率预测失败: {pf['error']}"

        return fig, summary_md, sig_df, prob_md, ""
    except Exception as e:
        import traceback
        return None, "❌ 分析失败", pd.DataFrame(), "", f"{e}\n\n{traceback.format_exc()[:500]}"


DISCLAIMER_MD = (
    "**⚠️ 仅供教育研究使用 / EDUCATIONAL & RESEARCH PURPOSE ONLY**\n\n"
    "本工具不预测未来价格、不提供投资建议、不推荐买卖任何证券。"
    "仅展示客观检测的技术事件、21 年回测的历史频率。"
    "**过去频率不能预测未来。** 作者非美国持牌投资顾问 (RIA)。所有交易决定为使用者自身责任。\n\n"
    "This tool does NOT predict future prices, NOT provide investment advice, "
    "NOT recommend buying/selling any security."
)


def build_ui():
    with gr.Blocks(title="技术指标历史频率分析 (Gradio)",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 股票技术指标历史频率分析 (Gradio 版)")
        gr.Markdown(DISCLAIMER_MD)

        with gr.Row():
            symbol_box = gr.Textbox(value="AAPL", label="股票代码",
                                    placeholder="例如: AAPL, TSLA, SPY")
            window_box = gr.Dropdown(
                ["自动 (最稳定)", "1 个月", "2 个月", "3 个月", "6 个月", "1 年", "2 年"],
                value="自动 (最稳定)", label="时间窗口",
            )
            pf_horizon = gr.Dropdown([5, 10, 20], value=10, label="概率预测期 (天)")
            pf_conf = gr.Dropdown(["80%", "90%", "95%"], value="90%", label="置信水平")

        with gr.Row():
            overlays_box = gr.CheckboxGroup(
                ["SMA50", "SMA200", "Bollinger", "Keltner",
                 "Supertrend", "Ichimoku", "VWAP"],
                value=["SMA50", "SMA200"],
                label="主图叠加",
            )
            subplots_box = gr.CheckboxGroup(
                ["Volume", "MACD", "RSI", "ADX", "TTM Mom"],
                value=["Volume", "MACD", "RSI"],
                label="副图",
            )

        analyze_btn = gr.Button("📥 加载并分析", variant="primary")

        with gr.Row():
            summary_out = gr.Markdown(label="摘要")
            prob_out = gr.Markdown(label="概率预测")
        chart_out = gr.Plot(label="K 线图")
        gr.Markdown("### 当前触发的信号 (按 Alpha 大小排序)")
        sig_out = gr.Dataframe(label="", wrap=True)
        error_out = gr.Textbox(label="错误信息", visible=True)

        analyze_btn.click(
            fn=analyze,
            inputs=[symbol_box, window_box, overlays_box, subplots_box, pf_horizon, pf_conf],
            outputs=[chart_out, summary_out, sig_out, prob_out, error_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # HF Spaces 用 0.0.0.0:7860 默认; 本地用 localhost
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)
