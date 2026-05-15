"""
股票技术指标历史频率分析工具
Stock Indicator Historical Frequency Viewer (Chinese UI + English Terms)
============================================================================
三个模式:
1. 单股分析 - 输入代码, 查看当前触发的信号 + 21年历史频率
2. 多周期共振 - 日/周/月 三个时间框架的 MACD/MA 状态
3. 股票池扫描 - 一次扫描多只股票

不预测、不推荐、不建议。
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

from extended_signals import compute_all_signals
from advanced_patterns import detect_all_advanced_patterns
from extra_patterns import detect_extra_patterns
from historical_stats import SIGNAL_STATS, BASELINE_10D, format_freq_text, stability_badge
from new_indicators import compute_new_indicators
from confluence import compute_confluence_summary, find_optimal_window
from plotly_charts import build_main_chart, DEFAULT_OVERLAYS, SUBPLOT_OPTIONS
import finnhub_data
from conformal import probability_forecast, conformalize_external_forecaster
import kronos_predict

st.set_page_config(
    page_title="技术指标频率分析",
    page_icon="📊",
    layout="wide",
)

# ============== 信号中文翻译 ==============
# Key 对应 historical_stats.py 里的 signal key
# Value 是中文显示名(保留英文术语在括号里)
LABEL_ZH = {
    # Cross signals
    "macd_gc": "MACD 金叉 (MACD Golden Cross)",
    "macd_dc": "MACD 死叉 (MACD Death Cross)",
    "ma_short_gc": "短期均线金叉 5/20 (Short MA Cross Up)",
    "ma_short_dc": "短期均线死叉 5/20 (Short MA Cross Down)",
    "ma_mid_gc": "中期均线金叉 20/50 (Mid MA Cross Up)",
    "ma_mid_dc": "中期均线死叉 20/50 (Mid MA Cross Down)",
    "ma_long_gc": "经典金叉 50/200 (Long MA Golden Cross)",
    "ma_long_dc": "经典死叉 50/200 (Long MA Death Cross)",
    "kdj_gc": "KDJ 金叉 (KDJ Cross Up)",
    "kdj_dc": "KDJ 死叉 (KDJ Cross Down)",
    "kdj_gc_oversold": "KDJ 超卖区金叉 (KDJ Cross Up in Oversold)",
    "kdj_dc_overbought": "KDJ 超买区死叉 (KDJ Cross Down in Overbought)",
    "dmi_gc": "DMI 金叉 +DI/-DI (DMI Cross Up)",
    "dmi_dc": "DMI 死叉 +DI/-DI (DMI Cross Down)",
    "dmi_gc_strong": "DMI 强趋势金叉 ADX>25 (DMI Cross Up Strong)",
    "dmi_dc_strong": "DMI 强趋势死叉 ADX>25 (DMI Cross Down Strong)",
    "psar_flip_up": "PSAR 翻转向上 (Parabolic SAR Flip Up)",
    "psar_flip_dn": "PSAR 翻转向下 (Parabolic SAR Flip Down)",

    # Support / Resistance
    "near_sma200": "接近 200 日均线 ±2% (Near 200-day MA)",
    "near_swing_high": "接近 60 日高点 ±2% (Near 60d Swing High)",
    "near_swing_low": "接近 60 日低点 ±2% (Near 60d Swing Low)",
    "near_bb_upper": "接近布林上轨 (Near Bollinger Upper)",
    "near_bb_lower": "接近布林下轨 (Near Bollinger Lower)",
    "near_r1": "接近枢轴 R1 (Near Pivot R1)",
    "near_s1": "接近枢轴 S1 (Near Pivot S1)",

    # Candle patterns
    "pat_doji": "十字星 (Doji)",
    "pat_hammer": "锤子线 (Hammer)",
    "pat_inverted_hammer": "倒锤子线 (Inverted Hammer)",
    "pat_bull_engulf": "看涨吞没 (Bullish Engulfing)",
    "pat_bear_engulf": "看跌吞没 (Bearish Engulfing)",
    "pat_three_white": "三阳兵 (Three White Soldiers)",
    "pat_three_black": "三阴兵 (Three Black Crows)",

    # Advanced chart patterns
    "pat_double_bottom": "双底 W (Double Bottom)",
    "pat_double_top": "双顶 M (Double Top)",
    "pat_head_shoulders": "头肩顶 (Head and Shoulders)",
    "pat_inverse_head_shoulders": "倒头肩 (Inverse Head and Shoulders)",
    "pat_uptrend_channel": "上升通道 R²>0.5 (Uptrend Channel)",
    "pat_downtrend_channel": "下降通道 R²>0.5 (Downtrend Channel)",
    "pat_horizontal_channel": "水平通道 (Horizontal Channel)",
    "pat_triangle_symmetric": "对称三角整理 (Symmetric Triangle)",
    "pat_triangle_ascending": "上升三角 (Ascending Triangle)",
    "pat_triangle_descending": "下降三角 (Descending Triangle)",
    "pat_breakout_up": "向上突破 20日新高 (Breakout Up)",
    "pat_breakout_down": "向下突破 20日新低 (Breakout Down)",
    "pat_morning_star": "早晨之星 (Morning Star)",
    "pat_evening_star": "黄昏之星 (Evening Star)",

    # Extra patterns
    "pat_rising_wedge": "上升楔形 (Rising Wedge)",
    "pat_falling_wedge": "下降楔形 (Falling Wedge)",
    "pat_bull_flag": "多头旗形 (Bull Flag)",
    "pat_bear_flag": "空头旗形 (Bear Flag)",
    "pat_pennant": "三角旗 (Pennant)",
    "pat_cup_handle": "杯柄形 (Cup and Handle)",
    "pat_rounding_bottom": "圆底 (Rounding Bottom)",
    "pat_gap_up": "跳空高开 >2% (Gap Up)",
    "pat_gap_down": "跳空低开 >2% (Gap Down)",
    "pat_inside_bar": "内包线 (Inside Bar)",
    "pat_outside_bar": "外包线 (Outside Bar)",

    # === 新指标信号 (new_indicators.py) ===
    "rsi_oversold": "RSI 超卖 <30 (RSI Oversold)",
    "rsi_overbought": "RSI 超买 >70 (RSI Overbought)",
    "rsi_cross_up_30": "RSI 从超卖区回升 (RSI Cross Up 30)",
    "rsi_cross_dn_70": "RSI 从超买区回落 (RSI Cross Down 70)",
    "rsi_bullish_div": "RSI 底背离 (RSI Bullish Divergence)",
    "rsi_bearish_div": "RSI 顶背离 (RSI Bearish Divergence)",
    "cci_oversold": "CCI 超卖 <-100 (CCI Oversold)",
    "cci_overbought": "CCI 超买 >100 (CCI Overbought)",
    "cci_cross_up_minus100": "CCI 从超卖区回升 (CCI Cross Up -100)",
    "cci_cross_dn_plus100": "CCI 从超买区回落 (CCI Cross Down +100)",
    "wr_oversold": "Williams %R 超卖 (WR Oversold)",
    "wr_overbought": "Williams %R 超买 (WR Overbought)",
    "wr_cross_up_minus80": "WR 从超卖区回升 (WR Cross Up -80)",
    "wr_cross_dn_minus20": "WR 从超买区回落 (WR Cross Down -20)",
    "obv_trend_up": "OBV 趋势向上 (OBV Trend Up)",
    "obv_trend_dn": "OBV 趋势向下 (OBV Trend Down)",
    "obv_bullish_div": "OBV 底背离 (OBV Bullish Divergence)",
    "obv_bearish_div": "OBV 顶背离 (OBV Bearish Divergence)",
    "atr_high_volatility": "ATR 异常高波动 (ATR High Volatility)",
    "atr_low_volatility": "ATR 异常低波动 (ATR Low Volatility)",
    "atr_expanding": "ATR 连续扩大 (ATR Expanding)",
    "atr_contracting": "ATR 连续收缩 (ATR Contracting)",
    "above_vwap": "价格在 VWAP 上方 (Above VWAP)",
    "below_vwap": "价格在 VWAP 下方 (Below VWAP)",
    "vwap_cross_up": "价格向上穿越 VWAP (VWAP Cross Up)",
    "vwap_cross_dn": "价格向下穿越 VWAP (VWAP Cross Down)",
    "supertrend_bullish": "Supertrend 多头 (Supertrend Bullish)",
    "supertrend_bearish": "Supertrend 空头 (Supertrend Bearish)",
    "supertrend_flip_up": "Supertrend 翻多 (Supertrend Flip Up)",
    "supertrend_flip_dn": "Supertrend 翻空 (Supertrend Flip Down)",
    "ichimoku_above_cloud": "价格在一目云上方 (Above Ichimoku Cloud)",
    "ichimoku_below_cloud": "价格在一目云下方 (Below Ichimoku Cloud)",
    "ichimoku_tk_cross_up": "一目转换线金叉 (Ichimoku TK Cross Up)",
    "ichimoku_tk_cross_dn": "一目转换线死叉 (Ichimoku TK Cross Down)",
    "ichimoku_cloud_bullish": "一目云层看涨 (Ichimoku Cloud Bullish)",
    "nx_bull_zone": "NX 多头排列 EMA5>20>60 (NX Bull Zone)",
    "nx_bear_zone": "NX 空头排列 EMA5<20<60 (NX Bear Zone)",
    "nx_flip_bull": "NX 翻多头 (NX Flip Bullish)",
    "nx_flip_bear": "NX 翻空头 (NX Flip Bearish)",
    "cd_bullish_div": "MACD DIF 底背离 (CD Bullish Divergence)",
    "cd_bearish_div": "MACD DIF 顶背离 (CD Bearish Divergence)",
    "cd_hist_bullish_div": "MACD 柱状底背离 (CD Hist Bullish Div)",
    "cd_hist_bearish_div": "MACD 柱状顶背离 (CD Hist Bearish Div)",
    "trix_cross_up": "TRIX 金叉信号线 (TRIX Cross Up)",
    "trix_cross_dn": "TRIX 死叉信号线 (TRIX Cross Down)",
    "trix_positive": "TRIX > 0 上涨动量 (TRIX Positive)",
    "trix_negative": "TRIX < 0 下跌动量 (TRIX Negative)",

    # === Keltner Channels ===
    "kc_above_upper": "突破 Keltner 上轨 (Above KC Upper)",
    "kc_below_lower": "跌破 Keltner 下轨 (Below KC Lower)",
    "kc_inside": "Keltner 通道内 (Inside KC)",
    "kc_break_up": "向上突破 Keltner 上轨 (KC Break Up)",
    "kc_break_dn": "向下突破 Keltner 下轨 (KC Break Down)",

    # === Chaikin Money Flow ===
    "cmf_positive": "CMF > 0 资金流入 (CMF Positive)",
    "cmf_negative": "CMF < 0 资金流出 (CMF Negative)",
    "cmf_strong_bull": "CMF > 0.25 强烈买盘 (CMF Strong Bull)",
    "cmf_strong_bear": "CMF < -0.25 强烈卖盘 (CMF Strong Bear)",
    "cmf_cross_zero_up": "CMF 上穿 0 (CMF Cross Zero Up)",
    "cmf_cross_zero_dn": "CMF 下穿 0 (CMF Cross Zero Down)",

    # === TTM Squeeze ===
    "squeeze_on": "TTM 挤压 (Squeeze On)",
    "squeeze_off": "TTM 释放 (Squeeze Off)",
    "squeeze_fired_up": "TTM 向上释放 (Squeeze Fired Up)",
    "squeeze_fired_dn": "TTM 向下释放 (Squeeze Fired Down)",

    # === Heikin-Ashi ===
    "ha_bullish": "HA 阳线 (HA Bullish)",
    "ha_bearish": "HA 阴线 (HA Bearish)",
    "ha_strong_bull": "HA 强势上升 3+ 根无下影 (HA Strong Bull)",
    "ha_strong_bear": "HA 强势下跌 3+ 根无上影 (HA Strong Bear)",
    "ha_doji": "HA 十字星 (HA Doji)",
    "ha_flip_bull": "HA 转涨 (HA Flip Bull)",
    "ha_flip_bear": "HA 转跌 (HA Flip Bear)",
}


def zh_label(sig_key, fallback=None):
    """获取信号的中文显示名"""
    return LABEL_ZH.get(sig_key, fallback or SIGNAL_STATS.get(sig_key, {}).get("label", sig_key))


# ============== 免责声明 ==============
st.error(
"""**⚠️ 仅供教育研究使用 / EDUCATIONAL & RESEARCH PURPOSE ONLY**

本工具**不预测未来价格**,**不提供投资建议**,**不推荐买卖任何证券**。
仅展示: 客观检测的技术事件、21 年回测的历史频率、公开财报数据(无评级)。

**过去频率不能预测未来。** 作者非美国持牌投资顾问 (RIA)。所有交易决定为使用者自身责任。

This tool does NOT predict future prices, NOT provide investment advice, NOT recommend buying/selling any security.
Past frequencies do not predict the future. Author is not a registered investment advisor. All trading decisions are 100% the user's responsibility.
""")

CACHE_DIR = Path.home() / ".freq_app_cache"
CACHE_DIR.mkdir(exist_ok=True)


@st.cache_data(ttl=900)
def fetch_data(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    cache = CACHE_DIR / f"{symbol}_{interval}_{start}_{end}.pkl"
    if cache.exists():
        try:
            cached = pd.read_pickle(cache)
            if not cached.empty and cached.iloc[-1]["timestamp"].date() >= (datetime.now(timezone.utc).date() - timedelta(days=1)):
                return cached
        except Exception:
            pass
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.to_pickle(cache)
        return df
    except Exception as e:
        st.error(f"数据获取失败 {symbol}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_earnings(symbol: str):
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        try:
            cal = t.calendar
        except Exception:
            cal = None
        return {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "next_earnings_date": cal.get("Earnings Date") if isinstance(cal, dict) else None,
            "long_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        return {}


def compute_macd_state(df, fast=12, slow=26, signal=9):
    if df.empty or len(df) < 47:
        return None, None
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_sig"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    last = df.iloc[-1]
    state = "bullish" if last["macd_hist"] > 0 else "bearish"
    return state, last["macd_hist"]


def compute_ma_state(df, short=50, long=200):
    if df.empty or len(df) < long:
        return None
    df = df.copy()
    df[f"sma{short}"] = df["close"].rolling(short).mean()
    df[f"sma{long}"] = df["close"].rolling(long).mean()
    last = df.iloc[-1]
    if pd.isna(last[f"sma{long}"]):
        return None
    return "bullish" if last[f"sma{short}"] > last[f"sma{long}"] else "bearish"


def stability_badge_zh(stat):
    """中文版稳定性描述"""
    if not stat:
        return ""
    test_pct = stat.get("test_pct_up_10d")
    pct = stat.get("pct_up_10d")
    if test_pct is None or pct is None:
        return "数据有限"
    drift = test_pct - pct
    if abs(drift) < 1.0:
        d = f"近 6 年与 21 年匹配 ({drift:+.1f}%)"
    elif abs(drift) < 3.0:
        d = f"近 6 年轻微漂移 ({drift:+.1f}%)"
    elif abs(drift) < 5.0:
        d = f"近 6 年显著漂移 ({drift:+.1f}%)"
    else:
        d = f"近 6 年大幅漂移 ({drift:+.1f}%)"
    oos = stat.get("oos_significant", False)
    o = "近 6 年仍显著 p<0.05" if oos else "近 6 年不显著 p>=0.05"
    return f"{d}; {o}"


@st.cache_data(ttl=900)
def fh_company_news(symbol: str, days_back: int = 7):
    return finnhub_data.get_company_news(symbol, days_back=days_back, limit=20)


@st.cache_data(ttl=1800)
def fh_news_sentiment(symbol: str):
    return finnhub_data.get_news_sentiment(symbol)


@st.cache_data(ttl=3600)
def fh_insider_tx(symbol: str):
    return finnhub_data.get_insider_transactions(symbol, limit=20)


@st.cache_data(ttl=3600)
def fh_recommendations(symbol: str):
    return finnhub_data.get_recommendations(symbol)


@st.cache_data(ttl=3600)
def fh_earnings_surprise(symbol: str):
    return finnhub_data.get_earnings_surprise(symbol)


def render_finnhub_section(sym: str):
    """渲染 Finnhub 数据面板。如未配置 API key, 显示如何获取的提示。"""
    api_key = finnhub_data.get_api_key()
    st.markdown("---")
    st.markdown("## 📰 Finnhub 数据 (新闻 / 情绪 / 内幕 / 评级)")

    if not api_key:
        st.info(
            "🔑 **未配置 Finnhub API key** — 此面板提供新闻、情绪、内幕交易、分析师评级数据。\n\n"
            "获取免费 key (60 次/分钟, 永久免费):\n"
            "1. 访问 https://finnhub.io 注册账户\n"
            "2. 在 Dashboard 复制 API key\n"
            "3. 设置环境变量: `export FINNHUB_API_KEY=<your-key>` 或在 HuggingFace Space "
            "Settings → Variables and secrets 添加 `FINNHUB_API_KEY`\n"
            "4. 重启应用"
        )
        return

    st.caption(
        "🔌 数据由 Finnhub.io 提供 (免费版 60 次/分钟)。"
        "**分析师推荐是第三方机构观点, 不是本工具的建议**。"
    )

    fh_tabs = st.tabs(["📰 公司新闻", "📊 新闻情绪", "👤 内幕交易", "🎯 分析师评级", "💰 财报惊喜"])

    # === 公司新闻 ===
    with fh_tabs[0]:
        with st.spinner("获取最近 7 天新闻..."):
            news = fh_company_news(sym, days_back=7)
        if not news["ok"]:
            st.warning(f"无法获取新闻: {news['error']}")
        elif not news["data"]:
            st.info("最近 7 天无相关新闻。")
        else:
            for item in news["data"][:10]:
                with st.container():
                    st.markdown(f"**{item['headline']}**")
                    st.caption(f"📅 {item['datetime']} | 📢 {item['source']} | 🏷️ {item['category']}")
                    if item["summary"]:
                        st.write(item["summary"])
                    if item["url"]:
                        st.caption(f"🔗 [原文链接]({item['url']})")
                    st.markdown("---")

    # === 新闻情绪 ===
    with fh_tabs[1]:
        with st.spinner("获取新闻情绪..."):
            sent = fh_news_sentiment(sym)
        if not sent["ok"]:
            st.warning(f"无法获取情绪数据: {sent['error']}")
        elif not sent["data"]:
            st.info("无情绪数据")
        else:
            d = sent["data"]
            sc1, sc2, sc3, sc4 = st.columns(4)
            score = d.get("company_news_score")
            sc1.metric("公司新闻情绪分 (0-1)",
                       f"{score:.2f}" if score is not None else "—",
                       help="越接近 1 = 整体新闻越正面 (Finnhub 计算, 仅供参考)")
            sector_avg = d.get("sector_avg_news_score")
            sc2.metric("行业平均",
                       f"{sector_avg:.2f}" if sector_avg is not None else "—")
            bull = d.get("bullish_pct")
            sc3.metric("看涨新闻占比",
                       f"{bull*100:.1f}%" if bull is not None else "—")
            bear = d.get("bearish_pct")
            sc4.metric("看跌新闻占比",
                       f"{bear*100:.1f}%" if bear is not None else "—")

            buzz = d.get("buzz")
            arts = d.get("articles_last_week")
            wkly = d.get("weekly_avg_articles")
            wkly_str = f"{wkly:.1f}" if wkly is not None else "—"
            buzz_str = f"{buzz:.2f}" if buzz is not None else "—"
            arts_str = str(arts) if arts is not None else "—"
            st.caption(
                f"📈 上周文章数: {arts_str} | 周均: {wkly_str} | Buzz (相对热度): {buzz_str}"
            )
            st.warning(
                "⚠️ Finnhub 情绪分基于自动 NLP 分析的新闻头条, 可能误读上下文。"
                "情绪分数 ≠ 价格预测, 不构成投资建议。"
            )

    # === 内幕交易 ===
    with fh_tabs[2]:
        with st.spinner("获取内幕交易..."):
            ins = fh_insider_tx(sym)
        if not ins["ok"]:
            st.warning(f"无法获取内幕数据: {ins['error']}")
        elif not ins["data"]:
            st.info("无内幕交易记录 (或公司无 SEC Form 4 报告)")
        else:
            tx_code_zh = {
                "P": "买入 (Purchase)", "S": "卖出 (Sale)",
                "A": "授予 (Award)", "M": "期权行权 (Option Exercise)",
                "G": "赠予 (Gift)", "F": "缴税转让 (Tax Withholding)",
                "D": "递交给发行人 (Disposition to Issuer)",
            }
            rows = []
            for t in ins["data"]:
                code = t.get("transactionCode", "")
                rows.append({
                    "申报日": t.get("filingDate", ""),
                    "交易日": t.get("transactionDate", ""),
                    "姓名": t.get("name", ""),
                    "代码": f"{code} ({tx_code_zh.get(code, '其他')})",
                    "股数变动": t.get("change") if t.get("change") is not None else t.get("share"),
                    "价格": f"${t['transactionPrice']:.2f}" if t.get("transactionPrice") else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("数据来源: SEC Form 4 通过 Finnhub 聚合。**只是历史申报, 不预示后市方向**。")

    # === 分析师评级 ===
    with fh_tabs[3]:
        with st.spinner("获取分析师评级..."):
            rec = fh_recommendations(sym)
        if not rec["ok"]:
            st.warning(f"无法获取评级: {rec['error']}")
        elif not rec["data"]:
            st.info("无分析师评级数据")
        else:
            st.error(
                "⚠️ **以下是华尔街第三方机构的评级汇总, 不是本工具的建议**。"
                "本工具不推荐买入/卖出, 评级仅作信息展示。"
            )
            rows = []
            for r in rec["data"]:
                total = r["strongBuy"] + r["buy"] + r["hold"] + r["sell"] + r["strongSell"]
                rows.append({
                    "月份": r["period"],
                    "强烈买入": r["strongBuy"],
                    "买入": r["buy"],
                    "持有": r["hold"],
                    "卖出": r["sell"],
                    "强烈卖出": r["strongSell"],
                    "总计": total,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # === 财报惊喜 ===
    with fh_tabs[4]:
        with st.spinner("获取财报历史..."):
            er = fh_earnings_surprise(sym)
        if not er["ok"]:
            st.warning(f"无法获取财报数据: {er['error']}")
        elif not er["data"]:
            st.info("无财报数据")
        else:
            rows = []
            for e in er["data"]:
                surp = e.get("surprisePercent")
                rows.append({
                    "季度": e.get("period", ""),
                    "实际 EPS": f"${e['actual']:.2f}" if e.get("actual") is not None else "—",
                    "预期 EPS": f"${e['estimate']:.2f}" if e.get("estimate") is not None else "—",
                    "惊喜 %": f"{surp:+.1f}%" if surp is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


@st.cache_data(ttl=3600)
def search_symbols(query: str, max_results: int = 10):
    """
    用 Yahoo Finance Search API 查找符号候选。
    支持: 股票、ETF、加密币、期货、外汇。
    限制: 不支持中文 (Yahoo API 限制), 请用英文/拼音 (例如 alibaba 而不是 阿里巴巴)。
    """
    import urllib.request
    import urllib.parse
    import json

    if not query or not query.strip():
        return []
    try:
        params = urllib.parse.urlencode({
            "q": query.strip(),
            "quotesCount": max_results,
            "newsCount": 0,
        })
        url = f"https://query1.finance.yahoo.com/v1/finance/search?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        quotes = data.get("quotes", [])
        results = []
        for q in quotes:
            sym = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or ""
            qtype = q.get("quoteType", "")
            exch = q.get("exchange", "")
            if sym:
                results.append({
                    "symbol": sym,
                    "name": name,
                    "type": qtype,
                    "exchange": exch,
                })
        return results
    except Exception:
        return []  # 静默失败, 不影响主功能


# ============== UI ==============
st.title("📊 股票技术指标历史频率分析工具")
st.caption("输入股票代码 → 查看当前触发的信号 + 它们在过去 21 年的历史频率。**不预测未来**。")

tab1, tab2, tab3 = st.tabs([
    "📈 单股分析",
    "🔄 多周期共振",
    "🔍 股票池扫描",
])

# ============== TAB 1: 单股分析 ==============
with tab1:
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        symbol_input = st.text_input("股票代码", value="AAPL", key="t1_sym",
                                      placeholder="例如: AAPL, TSLA, SPY, BABA")
    with col2:
        period = st.selectbox("时间窗口",
                              ["自动 (最稳定)", "1 个月", "2 个月", "3 个月", "6 个月", "1 年", "2 年"],
                              index=0, key="t1_period")
    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("📥 加载数据", type="primary", use_container_width=True, key="t1_btn")
    with col4:
        st.write("")
        st.write("")
        show_window_diag = st.checkbox("窗口诊断", value=False, key="t1_winddiag",
                                       help="显示各候选窗口的共振稳定性对比表")

    period_to_days = {"1 个月": 30, "2 个月": 60, "3 个月": 90,
                      "6 个月": 180, "1 年": 365, "2 年": 730}
    if period == "自动 (最稳定)":
        days = 365  # 暂取最大候选, 数据加载后用 find_optimal_window 重选
        auto_window = True
    else:
        days = period_to_days[period]
        auto_window = False
    st.caption(f"📡 数据源: yfinance (延迟 15-20 分钟) | 最后刷新: {datetime.now().strftime('%H:%M:%S')}")

    # === 符号搜索 (帮用户找代码) ===
    with st.expander("🔍 不知道代码? 在这里搜索 (Search by company name)"):
        st.caption("输入公司英文名或拼音 (例如 'apple', 'alibaba', 'tencent', 'bitcoin') → 显示候选代码 → 复制到上方输入框")
        st.caption("⚠️ Yahoo API 不支持中文搜索, 请用英文/拼音。支持: 股票、ETF、加密币、期货、外汇、各国交易所。")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input("搜索关键词", value="", key="t1_search",
                                          placeholder="例如: apple, tesla, alibaba, bitcoin, gold")
        with search_col2:
            st.write("")
            st.write("")
            do_search = st.button("🔍 搜索", key="t1_search_btn", use_container_width=True)

        if do_search and search_query.strip():
            with st.spinner("搜索中..."):
                results = search_symbols(search_query)
            if not results:
                st.warning(f"未找到 '{search_query}' 的候选。请试英文/拼音。")
            else:
                st.success(f"找到 {len(results)} 个候选 (点击代码复制到输入框上方):")
                # 表格化展示
                display_rows = []
                for r in results:
                    type_zh = {
                        "EQUITY": "股票",
                        "ETF": "ETF",
                        "CRYPTOCURRENCY": "加密币",
                        "FUTURE": "期货",
                        "CURRENCY": "外汇",
                        "INDEX": "指数",
                        "MUTUALFUND": "基金",
                    }.get(r["type"], r["type"])
                    display_rows.append({
                        "代码": r["symbol"],
                        "名称": r["name"][:50],
                        "类型": type_zh,
                        "交易所": r["exchange"],
                    })
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
                st.caption("💡 复制代码 (例如 AAPL, 9988.HK, BTC-USD) → 粘贴到上方 '股票代码' 输入框 → 点 '加载数据'")

    if analyze_btn:
        sym = symbol_input.strip().upper() if symbol_input else "AAPL"
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 250)

        with st.spinner(f"加载中 {sym}..."):
            d = fetch_data(sym, "1d", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            earnings = fetch_earnings(sym)

        if d.empty or len(d) < 250:
            st.error(f"❌ 无数据或历史不足: {sym} (需要至少 250 个交易日)")
        else:
            df_full = compute_all_signals(d.copy())
            df_full = detect_all_advanced_patterns(df_full)
            df_full = detect_extra_patterns(df_full)
            df_full = compute_new_indicators(df_full)

            # === Task 4: 自动最优窗口 ===
            window_diag = None
            if auto_window:
                window_diag = find_optimal_window(
                    df_full,
                    candidate_windows=(30, 60, 90, 180, 365),
                )
                days = window_diag["best_window"]
                st.success(
                    f"🤖 自动选择最佳窗口: **{days} 天** — {window_diag['best_reason']}"
                )
            elif show_window_diag:
                window_diag = find_optimal_window(
                    df_full,
                    candidate_windows=(30, 60, 90, 180, 365),
                )

            if window_diag and window_diag["candidates"] and show_window_diag:
                with st.expander("📊 窗口稳定性诊断 (Window Stability Diagnostics)"):
                    st.caption(
                        "比较各候选窗口内 *全部* bar 的共振评分稳定性。"
                        "符号翻转率 (flip_rate) 越低 → 多空切换越少 → 信号方向越稳定。"
                        "std 是评分的标准差, 越小波动越小。"
                    )
                    rows = []
                    for c in window_diag["candidates"]:
                        rows.append({
                            "窗口 (天)": c["window"],
                            "Bar 数": c["n_bars"],
                            "翻转率 %": f"{c['flip_rate']*100:.1f}%",
                            "符号翻转": c["sign_flips"],
                            "共振 std": c["std"],
                            "共振均值": c["mean"],
                            "最新评分": c["last"],
                            "极差": c["range"],
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption(
                        f"🏆 推荐窗口: **{window_diag['best_window']} 天** "
                        f"— {window_diag['best_reason']}"
                    )

            display_start = end - timedelta(days=days)
            # 修复: display_start 已经是 tz-aware (因为 end 用 datetime.now(timezone.utc))
            # 不能再传 tz="UTC" 参数, 直接 pd.Timestamp() 即可
            disp = df_full[df_full["timestamp"] >= pd.Timestamp(display_start)].copy()

            if disp.empty:
                st.error("选择的窗口数据不足")
            else:
                last = df_full.iloc[-1]
                prev = df_full.iloc[-2]

                st.markdown(f"## {sym}")
                if earnings.get("long_name"):
                    sector = earnings.get("sector") or "N/A"
                    industry = earnings.get("industry") or "N/A"
                    st.caption(f"**{earnings['long_name']}** — {sector} / {industry}")

                col_a, col_b, col_c, col_d = st.columns(4)
                change_pct = (last['close'] / prev['close'] - 1) * 100
                col_a.metric("当前价格", f"${last['close']:.2f}", f"{change_pct:+.2f}%")
                col_b.metric("MACD 状态",
                             "零轴上方" if last.get("macd_state") == 1
                             else "零轴下方" if last.get("macd_state") == -1
                             else "—")
                col_c.metric("ADX 趋势强度",
                             f"{last['adx']:.1f}" if pd.notna(last.get("adx")) else "—")
                col_d.metric("200 日均线之上",
                             "是" if last.get("above_sma200") else "否")


                # === 综合共振评分 ===
                st.markdown("---")
                conf = compute_confluence_summary(df_full)
                st.markdown(f"## 🎯 综合共振评分: {conf['score']:.0f}/100 → {conf['label']}")
                st.caption(conf['detail'])

                dim_names_zh = {"trend":"趋势","momentum":"动量","overbought_oversold":"超买超卖",
                                "volume":"成交量","volatility":"波动率","pattern":"形态"}
                conf_cols = st.columns(6)
                for idx, (dim_name, dim) in enumerate(conf['dimensions'].items()):
                    with conf_cols[idx]:
                        dim_zh = dim_names_zh.get(dim_name, dim_name)
                        color = "🟢" if dim['score'] > 0.3 else "🔴" if dim['score'] < -0.3 else "🟡"
                        st.metric(f"{color} {dim_zh}", f"{dim['score']:+.1f}")
                        st.caption(f"{dim.get('n_bull',0)}多 {dim.get('n_bear',0)}空 {dim.get('n_neutral',0)}中")

                st.warning(conf['disclaimer'])

                # === 新指标快照 ===
                st.markdown("## 📊 关键指标快照")
                snap_cols = st.columns(5)
                rsi_val = last.get('rsi')
                snap_cols[0].metric("RSI (14)", f"{rsi_val:.1f}" if pd.notna(rsi_val) else "—",
                                    "超买" if pd.notna(rsi_val) and rsi_val > 70 else "超卖" if pd.notna(rsi_val) and rsi_val < 30 else "中性",
                                    delta_color="off")
                cci_val = last.get('cci')
                snap_cols[1].metric("CCI (20)", f"{cci_val:.0f}" if pd.notna(cci_val) else "—")
                wr_val = last.get('williams_r')
                snap_cols[2].metric("Williams %R", f"{wr_val:.1f}" if pd.notna(wr_val) else "—")
                snap_cols[3].metric("Supertrend", "多头 ↑" if last.get('supertrend_bullish') else "空头 ↓")
                snap_cols[4].metric("NX 通道", "多头区" if last.get('nx_bull_zone') else "空头区" if last.get('nx_bear_zone') else "中性")

                snap_cols2 = st.columns(5)
                atr_val = last.get('atr')
                snap_cols2[0].metric("ATR (14)", f"{atr_val:.2f}" if pd.notna(atr_val) else "—")
                vwap_val = last.get('vwap')
                snap_cols2[1].metric("VWAP (20)", f"${vwap_val:.2f}" if pd.notna(vwap_val) else "—",
                                     "上方" if last.get('above_vwap') else "下方")
                snap_cols2[2].metric("Ichimoku", "云上 ↑" if last.get('ichimoku_above_cloud') else "云下 ↓" if last.get('ichimoku_below_cloud') else "云中")
                trix_val = last.get('trix')
                snap_cols2[3].metric("TRIX", f"{trix_val:.4f}" if pd.notna(trix_val) else "—",
                                     "正" if last.get('trix_positive') else "负" if last.get('trix_negative') else "—")
                # 背离信号
                divs = []
                if last.get('cd_bullish_div'): divs.append("MACD底背离")
                if last.get('cd_bearish_div'): divs.append("MACD顶背离")
                if last.get('rsi_bullish_div'): divs.append("RSI底背离")
                if last.get('rsi_bearish_div'): divs.append("RSI顶背离")
                if last.get('obv_bullish_div'): divs.append("OBV底背离")
                if last.get('obv_bearish_div'): divs.append("OBV顶背离")
                snap_cols2[4].metric("背离信号", f"{len(divs)} 个" if divs else "无")
                if divs:
                    st.info("🔄 检测到背离: " + ", ".join(divs))

                # 第三排: 新增指标 (CMF / Keltner / TTM Squeeze / Heikin-Ashi)
                snap_cols3 = st.columns(5)
                cmf_val = last.get("cmf")
                if pd.notna(cmf_val):
                    if cmf_val > 0.25:
                        cmf_state = "强烈买盘"
                    elif cmf_val < -0.25:
                        cmf_state = "强烈卖盘"
                    elif cmf_val > 0:
                        cmf_state = "买盘"
                    elif cmf_val < 0:
                        cmf_state = "卖盘"
                    else:
                        cmf_state = "中性"
                    snap_cols3[0].metric("CMF (20)", f"{cmf_val:+.3f}", cmf_state, delta_color="off")
                else:
                    snap_cols3[0].metric("CMF (20)", "—")

                # Keltner 位置
                kc_state = "通道内"
                if last.get("kc_above_upper"):
                    kc_state = "上轨之上 ↑"
                elif last.get("kc_below_lower"):
                    kc_state = "下轨之下 ↓"
                kc_mid_val = last.get("kc_middle")
                snap_cols3[1].metric("Keltner",
                                     f"${kc_mid_val:.2f}" if pd.notna(kc_mid_val) else "—",
                                     kc_state, delta_color="off")

                # TTM Squeeze
                if last.get("squeeze_fired_up"):
                    sq_state = "向上释放 ⚡↑"
                elif last.get("squeeze_fired_dn"):
                    sq_state = "向下释放 ⚡↓"
                elif last.get("squeeze_on"):
                    sq_state = "挤压中 (低波动)"
                else:
                    sq_state = "释放中"
                sq_mom = last.get("squeeze_momentum")
                snap_cols3[2].metric("TTM Squeeze",
                                     f"动量 {sq_mom:+.3f}" if pd.notna(sq_mom) else "—",
                                     sq_state, delta_color="off")

                # Heikin-Ashi
                if last.get("ha_strong_bull"):
                    ha_state = "强势上升"
                elif last.get("ha_strong_bear"):
                    ha_state = "强势下跌"
                elif last.get("ha_bullish"):
                    ha_state = "阳线"
                elif last.get("ha_bearish"):
                    ha_state = "阴线"
                else:
                    ha_state = "—"
                snap_cols3[3].metric("Heikin-Ashi", ha_state)

                # 新指标提示
                hints = []
                if last.get("squeeze_on"):
                    hints.append("TTM 挤压中")
                if last.get("squeeze_fired_up"):
                    hints.append("TTM 向上释放")
                if last.get("squeeze_fired_dn"):
                    hints.append("TTM 向下释放")
                if last.get("kc_break_up"):
                    hints.append("KC 上轨突破")
                if last.get("kc_break_dn"):
                    hints.append("KC 下轨突破")
                snap_cols3[4].metric("特殊事件", f"{len(hints)} 个" if hints else "无")
                if hints:
                    st.caption("⚡ " + ", ".join(hints))

                # === 概率预测 (Conformal Prediction) ===
                st.markdown("---")
                st.markdown("## 🎲 概率预测 (实验性 / 校准区间)")
                st.caption(
                    "用 AR(5) 对数收益率模型 + Split Conformal 校准, 给出未来 N 个交易日的"
                    "**点预测 + 置信区间 + 上涨概率**。Conformal 提供分布无关的覆盖率保证。"
                )
                pf_col_a, pf_col_b, pf_col_c = st.columns([1, 1, 2])
                with pf_col_a:
                    pf_horizon = st.selectbox("预测期 (交易日)", [5, 10, 20], index=1, key="t1_pf_h")
                with pf_col_b:
                    pf_conf = st.selectbox("置信水平",
                                           ["80% (alpha=0.20)", "90% (alpha=0.10)", "95% (alpha=0.05)"],
                                           index=1, key="t1_pf_conf")
                pf_alpha = {"80% (alpha=0.20)": 0.20, "90% (alpha=0.10)": 0.10, "95% (alpha=0.05)": 0.05}[pf_conf]

                with st.spinner("计算 conformal 校准..."):
                    pf = probability_forecast(df_full, horizon=pf_horizon, alpha=pf_alpha)

                if not pf["ok"]:
                    st.warning(f"无法计算: {pf['error']}")
                else:
                    pfm1, pfm2, pfm3, pfm4 = st.columns(4)
                    pfm1.metric(f"点预测 ({pf_horizon}d 收益率)",
                                f"{pf['point_pct']:+.2f}%",
                                delta_color="off")
                    pfm2.metric(f"{int((1-pf_alpha)*100)}% 置信区间",
                                f"[{pf['lower_pct']:+.1f}%, {pf['upper_pct']:+.1f}%]")
                    pfm3.metric("P(上涨)",
                                f"{pf['p_up']*100:.1f}%",
                                f"基线 {pf['naive_p_up_baseline']*100:.1f}%",
                                delta_color="off")
                    pfm4.metric(f"P(>+5%)  /  P(<-5%)",
                                f"{pf['p_up_5pct']*100:.1f}% / {pf['p_dn_5pct']*100:.1f}%")

                    cov = pf['calibration_coverage']
                    nominal = 1 - pf_alpha
                    cov_diff = cov - nominal
                    if abs(cov_diff) < 0.03:
                        cov_state = "✅ 校准良好"
                    elif abs(cov_diff) < 0.06:
                        cov_state = "⚠️ 校准轻微偏差"
                    else:
                        cov_state = "❌ 校准明显偏差 (近期波动率不稳定)"
                    st.caption(
                        f"**校准检查**: 校准集 n={pf['calibration_n']}, 实测覆盖率 = "
                        f"{cov*100:.1f}% (名义 {nominal*100:.0f}%) → {cov_state}。"
                        f" 方法: {pf['method']}。"
                    )
                    st.error(
                        "⚠️ **概率预测重要免责**: \n\n"
                        "• 这是一个简单 AR(5) 基线模型, **预测精度通常很差** "
                        "(AR 在金融时间序列上方差解释力 < 5%, 这是公开学术共识)\n\n"
                        "• Conformal 区间假设 *exchangeability*, 时间序列违反此假设, "
                        "实际覆盖率可能略低于名义值\n\n"
                        "• 校准基于过去 100 个滚动窗口, 无法预知 regime change / 黑天鹅事件\n\n"
                        "• **不构成投资建议, 不预测真实未来价格**, 仅供方法论展示"
                    )

                # === AI 概率预测 (Chronos / Kronos, 实验性) ===
                st.markdown("---")
                st.markdown("## 🧠 AI 概率预测 (Chronos-Bolt-Tiny + Conformal)")
                st.caption(
                    "用 Amazon Chronos-Bolt-Tiny (~9M 参数, Apache 2.0) 预测未来 OHLC, "
                    "原生输出 5 分位数; 同时叠加 Split Conformal 校准提供保证覆盖率的区间。"
                    "**首次运行需下载模型 (~40MB), CPU 推理 5-30 秒**。"
                )
                if not kronos_predict.is_available():
                    st.warning(
                        "🔧 AI 预测未启用 — 需要安装 ML 依赖:\n\n"
                        f"```\n{kronos_predict.get_import_error()}\n```\n\n"
                        "在 HuggingFace Spaces, 把 `chronos-forecasting` 和 "
                        "`torch --index-url https://download.pytorch.org/whl/cpu` 加到 requirements.txt 即可。"
                    )
                else:
                    ai_col1, ai_col2, ai_col3 = st.columns([1, 1, 2])
                    with ai_col1:
                        ai_horizon = st.selectbox("AI 预测期", [5, 10, 20], index=1, key="t1_ai_h")
                    with ai_col2:
                        ai_with_conformal = st.checkbox("叠加 Conformal 校准", value=True, key="t1_ai_conf",
                                                        help="勾选: 在 AI 预测上滚动校准 50 个窗口, 给出 90% CI (慢但稳)")
                    with ai_col3:
                        st.write("")
                        st.write("")
                        run_ai = st.button("🚀 运行 AI 预测", type="primary", key="t1_ai_btn")

                    if run_ai:
                        close_array = df_full["close"].astype(float).values
                        # Step 1: 原生 Chronos 概率分布
                        with st.spinner("Chronos 推理中 (首次需下载模型, 请耐心)..."):
                            chronos_out = kronos_predict.predict_distribution(
                                close_array, horizon=ai_horizon
                            )
                        if not chronos_out["ok"]:
                            st.error(f"AI 预测失败: {chronos_out['error']}")
                        else:
                            st.markdown(f"### 🤖 Chronos 原生分布 (模型: `{chronos_out['model']}`)")
                            ac1, ac2, ac3, ac4 = st.columns(4)
                            ac1.metric(f"中位数 ({ai_horizon}d)",
                                       f"{chronos_out['point_pct']:+.2f}%",
                                       delta_color="off")
                            qpct = chronos_out["quantiles_pct"]
                            if 0.1 in qpct and 0.9 in qpct:
                                ac2.metric("80% 分位数区间",
                                           f"[{qpct[0.1]:+.1f}%, {qpct[0.9]:+.1f}%]")
                            if chronos_out.get("p_up_native") is not None:
                                ac3.metric("AI: P(上涨)",
                                           f"{chronos_out['p_up_native']*100:.1f}%",
                                           f"基于 {chronos_out['n_samples']} 样本"
                                           if chronos_out['n_samples'] else "近似",
                                           delta_color="off")
                            if chronos_out.get("p_up_5pct_native") is not None and chronos_out.get("p_dn_5pct_native") is not None:
                                ac4.metric("AI: P(>+5%) / P(<-5%)",
                                           f"{chronos_out['p_up_5pct_native']*100:.1f}% / "
                                           f"{chronos_out['p_dn_5pct_native']*100:.1f}%")

                            # 显示完整分位数
                            if qpct:
                                qrows = [{"分位数": f"P{int(q*100)}",
                                          "log return": chronos_out["quantiles_log"].get(q),
                                          "% return": f"{qpct.get(q):+.2f}%" if qpct.get(q) is not None else "—"}
                                         for q in sorted(qpct.keys())]
                                st.dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)

                            # Step 2: 可选 Conformal 校准
                            if ai_with_conformal:
                                st.markdown("### 🎯 + Conformal 校准 (90% CI)")
                                with st.spinner(
                                    "滚动校准 (会运行 ~50 次 Chronos 推理, 可能需要 1-3 分钟)..."
                                ):
                                    conf_out = conformalize_external_forecaster(
                                        df_full,
                                        kronos_predict.chronos_point_forecaster,
                                        horizon=ai_horizon,
                                        alpha=0.10,
                                        cal_size=50,
                                    )
                                if not conf_out["ok"]:
                                    st.warning(f"Conformal 校准失败: {conf_out['error']}")
                                else:
                                    cc1, cc2, cc3, cc4 = st.columns(4)
                                    cc1.metric("点预测 (AI)",
                                               f"{conf_out['point_pct']:+.2f}%",
                                               delta_color="off")
                                    cc2.metric("90% Conformal CI",
                                               f"[{conf_out['lower_pct']:+.1f}%, {conf_out['upper_pct']:+.1f}%]")
                                    cc3.metric("Conformal P(上涨)",
                                               f"{conf_out['p_up']*100:.1f}%",
                                               f"基线 {conf_out['naive_p_up_baseline']*100:.1f}%",
                                               delta_color="off")
                                    cc4.metric("校准覆盖",
                                               f"{conf_out['calibration_coverage']*100:.1f}%",
                                               f"目标 90.0%",
                                               delta_color="off")
                                    st.caption(
                                        f"校准集 n={conf_out['calibration_n']}; "
                                        f"方法: {conf_out['method']}"
                                    )

                            st.error(
                                "⚠️ **AI 预测重要免责**:\n\n"
                                "• Chronos-Bolt 是通用时间序列模型, *未在金融数据上特调*, "
                                "对股票预测精度有限\n\n"
                                "• 模型使用过去 256 根 bar 作为上下文, "
                                "无法预知财报、宏观、地缘事件\n\n"
                                "• 概率分布反映模型的不确定性, *不是真实未来概率*\n\n"
                                "• Conformal 校准只在数据分布稳定时给出有效覆盖, "
                                "regime change 时失效\n\n"
                                "• **不构成投资建议, 不预测真实未来价格**"
                            )

                # === 当前触发的信号 ===
                st.markdown("---")
                st.markdown("## 🎯 当前触发的技术信号")
                st.caption(
                    "每个信号显示其 21 年历史频率、近 6 年频率、以及频率稳定性描述。"
                    "**历史频率不是预测**。"
                )

                detected = []
                for sig_col in SIGNAL_STATS.keys():
                    if sig_col in disp.columns:
                        val = last.get(sig_col)
                        if isinstance(val, (bool, np.bool_)) and bool(val):
                            detected.append(sig_col)

                if detected:
                    # 排序: 优先 oos_significant=True 的, 然后按 |alpha|
                    detected.sort(key=lambda s: (
                        not SIGNAL_STATS[s].get("oos_significant", False),
                        -abs(SIGNAL_STATS[s].get("alpha_10d", 0))
                    ))
                    for sig_col in detected:
                        stat = SIGNAL_STATS[sig_col]
                        with st.container():
                            st.markdown(f"### ✓ {zh_label(sig_col)}")
                            st.markdown(f"**频率稳定性 (TRAIN/TEST)**: {stability_badge_zh(stat)}")

                            col_l, col_r = st.columns([1, 2])
                            with col_l:
                                st.metric("历史触发次数", f"{stat['n']:,}")
                                st.metric("21 年后 10 日上涨率",
                                          f"{stat['pct_up_10d']:.1f}%",
                                          f"基准 {BASELINE_10D['pct_up_10d']:.1f}%",
                                          delta_color="off")
                                test_pct = stat.get("test_pct_up_10d", stat['pct_up_10d'])
                                delta_recent = test_pct - stat['pct_up_10d']
                                st.metric("近 6 年后 10 日上涨率",
                                          f"{test_pct:.1f}%",
                                          f"{delta_recent:+.1f}% vs 21年",
                                          delta_color="inverse")
                            with col_r:
                                sig_full = "✓ 统计显著" if stat["significant"] else "✗ 不显著"
                                sig_oos = "✓ 近 6 年仍显著" if stat.get("oos_significant") else "✗ 近 6 年不显著"
                                st.write(f"**21 年统计显著性**: {sig_full} (p = {stat['p_value']:.3f})")
                                st.write(f"**样本外验证 (2020-2026)**: {sig_oos}")
                                if "note" in stat:
                                    st.info(f"📝 备注: {stat['note']}")
                            st.markdown("---")
                else:
                    st.info("该股票今日无主要信号触发。")

                # === 财报 ===
                st.markdown("## 💼 财报信息")
                st.caption("公开财务数据。无评级、无判断。")
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                col_e1.metric("过去 P/E", f"{earnings['trailing_pe']:.2f}" if earnings.get("trailing_pe") else "N/A")
                col_e2.metric("未来 P/E", f"{earnings['forward_pe']:.2f}" if earnings.get("forward_pe") else "N/A")
                col_e3.metric("过去 EPS", f"${earnings['trailing_eps']:.2f}" if earnings.get("trailing_eps") else "N/A")
                if earnings.get("market_cap"):
                    mc = earnings["market_cap"]
                    if mc >= 1e12:
                        col_e4.metric("市值", f"${mc/1e12:.2f}T")
                    elif mc >= 1e9:
                        col_e4.metric("市值", f"${mc/1e9:.2f}B")
                    else:
                        col_e4.metric("市值", f"${mc/1e6:.1f}M")
                else:
                    col_e4.metric("市值", "N/A")

                col_g1, col_g2 = st.columns(2)
                if earnings.get("revenue_growth") is not None:
                    col_g1.metric("营收同比增长", f"{earnings['revenue_growth']*100:+.1f}%")
                if earnings.get("earnings_growth") is not None:
                    col_g2.metric("盈利同比增长", f"{earnings['earnings_growth']*100:+.1f}%")
                if earnings.get("next_earnings_date"):
                    ned = earnings["next_earnings_date"]
                    if isinstance(ned, list) and ned:
                        ned_str = str(ned[0])
                    else:
                        ned_str = str(ned)
                    st.info(f"📅 下次财报日期: {ned_str}")

                # === 图表 (Plotly 交互式) ===
                st.markdown("## 📈 交互式价格图表")
                st.caption(
                    "💡 拖动可平移 / 滚轮可缩放 / 双击复原 / 点击图例可隐藏/显示线条。"
                    "选择叠加项和副图后点 '重绘图表' 应用。"
                )

                with st.expander("⚙️ 图表设置 (Chart Settings)", expanded=True):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    with cc1:
                        st.markdown("**主图叠加 (Main Overlays)**")
                        ov_sma50 = st.checkbox("SMA50", value=True, key="t1_ov_sma50")
                        ov_sma200 = st.checkbox("SMA200", value=True, key="t1_ov_sma200")
                        ov_ema20 = st.checkbox("EMA20", value=False, key="t1_ov_ema20")
                    with cc2:
                        st.markdown("**通道 (Bands)**")
                        ov_bb = st.checkbox("Bollinger 20/2σ", value=False, key="t1_ov_bb")
                        ov_kc = st.checkbox("Keltner 20/2 ATR", value=False, key="t1_ov_kc")
                        ov_vwap = st.checkbox("VWAP(20)", value=False, key="t1_ov_vwap")
                    with cc3:
                        st.markdown("**趋势 (Trend)**")
                        ov_super = st.checkbox("Supertrend", value=False, key="t1_ov_super")
                        ov_ichi = st.checkbox("Ichimoku Cloud", value=False, key="t1_ov_ichi")
                        ov_markers = st.checkbox("信号标记 (Signal markers)", value=True, key="t1_ov_markers")
                    with cc4:
                        st.markdown("**副图 (Subplots)**")
                        sp_volume = st.checkbox("成交量 Volume", value=True, key="t1_sp_vol")
                        sp_macd = st.checkbox("MACD", value=True, key="t1_sp_macd")
                        sp_rsi = st.checkbox("RSI", value=True, key="t1_sp_rsi")
                        sp_adx = st.checkbox("ADX/DMI", value=False, key="t1_sp_adx")
                        sp_ttm = st.checkbox("TTM 动量", value=False, key="t1_sp_ttm")

                overlays = {
                    "sma50": ov_sma50, "sma200": ov_sma200, "ema20": ov_ema20,
                    "bollinger": ov_bb, "keltner": ov_kc, "vwap": ov_vwap,
                    "supertrend": ov_super, "ichimoku": ov_ichi,
                }
                subplots_selected = []
                if sp_volume: subplots_selected.append("volume")
                if sp_macd: subplots_selected.append("macd")
                if sp_rsi: subplots_selected.append("rsi")
                if sp_adx: subplots_selected.append("adx")
                if sp_ttm: subplots_selected.append("ttm_momentum")

                # 高度: 主图 500 + 每个副图 ~150
                chart_height = 500 + 150 * len(subplots_selected)

                fig = build_main_chart(
                    disp, sym,
                    overlays=overlays,
                    subplots=subplots_selected,
                    show_signal_markers=ov_markers,
                    height=chart_height,
                )
                st.plotly_chart(fig, use_container_width=True, theme=None,
                                config={"scrollZoom": True, "displaylogo": False,
                                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

                # === Finnhub 数据面板 (新闻 / 情绪 / 内幕 / 评级) ===
                render_finnhub_section(sym)

                # === 参考表 ===
                st.markdown("---")
                st.markdown(f"## 📚 完整参考表: 全部 {len(SIGNAL_STATS)} 个信号 (21 年回测验证)")
                st.info(
                    "**样本外验证**: 训练集 2005-2019 vs 测试集 2020-2026。"
                    "平均频率漂移 = 2.15%。部分信号在测试期 p>=0.05。"
                    "请用 '频率稳定性' 列查看每个信号在新旧数据间的漂移。"
                )
                ref_rows = []
                for k, stat in SIGNAL_STATS.items():
                    ref_rows.append({
                        "信号 (Signal)": zh_label(k),
                        "类别": stat["category"],
                        "样本数 n": f"{stat['n']:,}",
                        "21 年上涨率": f"{stat['pct_up_10d']:.1f}%",
                        "近 6 年上涨率": f"{stat.get('test_pct_up_10d', 0):.1f}%",
                        "Alpha": f"{stat['alpha_10d']:+.2f}%",
                        "p 值": f"{stat['p_value']:.3f}",
                        "频率稳定性": stability_badge_zh(stat),
                    })
                ref_df = pd.DataFrame(ref_rows)
                st.dataframe(ref_df, use_container_width=True, hide_index=True, height=600)


# ============== TAB 2: 多周期共振 ==============
with tab2:
    st.markdown("### 🔄 多周期共振分析")
    st.caption(
        "显示日线/周线/月线 三个时间框架下的 MACD 和 50/200 均线状态。"
        "当多个时间框架方向一致时,称为'共振'。**这只是描述,不是预测**。"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        sym2 = st.text_input("股票代码", value="AAPL", key="t2_sym",
                              placeholder="例如: AAPL, TSLA, SPY")
    with col2:
        st.write("")
        st.write("")
        btn2 = st.button("🔄 分析", type="primary", use_container_width=True, key="t2_btn")

    if btn2:
        sym2 = sym2.strip().upper()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365 * 10)

        with st.spinner(f"加载 {sym2} 多周期数据..."):
            d = fetch_data(sym2, "1d", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            w = fetch_data(sym2, "1wk", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            m = fetch_data(sym2, "1mo", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        if d.empty:
            st.error(f"无数据: {sym2}")
        else:
            states = []
            for tf, df in [("日线", d), ("周线", w), ("月线", m)]:
                if df.empty or len(df) < 50:
                    states.append({"timeframe": tf, "macd": "—", "macd_hist": "—", "ma_state": "—"})
                    continue
                macd_state, macd_hist = compute_macd_state(df)
                ma_state = compute_ma_state(df)
                states.append({
                    "timeframe": tf,
                    "macd": macd_state if macd_state else "—",
                    "macd_hist": f"{macd_hist:+.4f}" if macd_hist is not None else "—",
                    "ma_state": ma_state if ma_state else "—",
                    "last_close": df.iloc[-1]["close"] if not df.empty else None,
                })

            st.markdown(f"### {sym2} 多周期快照")
            cols = st.columns(3)
            for i, s in enumerate(states):
                with cols[i]:
                    st.markdown(f"#### {s['timeframe']}")
                    if s["macd"] == "bullish":
                        st.info(f"MACD: ↑ 零轴上方 (柱状 {s['macd_hist']})")
                    elif s["macd"] == "bearish":
                        st.info(f"MACD: ↓ 零轴下方 (柱状 {s['macd_hist']})")
                    else:
                        st.write(f"MACD: {s['macd']}")
                    if s["ma_state"] == "bullish":
                        st.info("50/200 均线: ↑ 多头排列")
                    elif s["ma_state"] == "bearish":
                        st.info("50/200 均线: ↓ 空头排列")
                    else:
                        st.write(f"50/200 均线: {s['ma_state']}")
                    if s.get("last_close"):
                        st.caption(f"最新收盘: ${s['last_close']:.2f}")

            macd_states = [s["macd"] for s in states if s["macd"] in ("bullish", "bearish")]
            ma_states = [s["ma_state"] for s in states if s["ma_state"] in ("bullish", "bearish")]

            st.markdown("---")
            st.markdown("### 📋 共振摘要")
            macd_aligned_bull = all(m == "bullish" for m in macd_states) and len(macd_states) >= 2
            macd_aligned_bear = all(m == "bearish" for m in macd_states) and len(macd_states) >= 2
            ma_aligned_bull = all(m == "bullish" for m in ma_states) and len(ma_states) >= 2
            ma_aligned_bear = all(m == "bearish" for m in ma_states) and len(ma_states) >= 2

            cols_a = st.columns(2)
            with cols_a[0]:
                st.markdown("**MACD 多周期共振**:")
                if macd_aligned_bull:
                    st.info(f"状态: {len(macd_states)} 个周期 MACD 均在零轴上方")
                elif macd_aligned_bear:
                    st.info(f"状态: {len(macd_states)} 个周期 MACD 均在零轴下方")
                else:
                    st.info("混合: 各周期 MACD 方向不一致")
            with cols_a[1]:
                st.markdown("**50/200 均线共振**:")
                if ma_aligned_bull:
                    st.info(f"状态: {len(ma_states)} 个周期 50 均线均在 200 均线上方")
                elif ma_aligned_bear:
                    st.info(f"状态: {len(ma_states)} 个周期 50 均线均在 200 均线下方")
                else:
                    st.info("混合: 各周期方向不一致")

            st.markdown("---")
            st.warning(
                "**诚实声明**: 共振是描述性观察, 不是预测性指标。"
                "在我们 21 年回测中, 多指标共振组合**没有**提升预测力。"
                "请把共振视为上下文信息, 不是买卖建议。"
            )


# ============== TAB 3: 股票池扫描 ==============
with tab3:
    st.markdown("### 🔍 股票池扫描")
    st.caption("输入多只股票代码用逗号分隔。扫描器显示每只股票当前触发了哪些信号。")

    PRESET_POOLS = {
        "美股科技七巨头": "AAPL,MSFT,NVDA,GOOGL,META,AMZN,TSLA",
        "美股银行": "JPM,BAC,WFC,GS,MS,C",
        "美股指数 ETF": "SPY,QQQ,IWM,DIA,VTI",
        "自定义": "",
    }
    preset = st.selectbox("预设股票池", list(PRESET_POOLS.keys()), key="t3_preset")
    if preset != "自定义":
        symbols_input = st.text_input("股票代码列表", value=PRESET_POOLS[preset], key="t3_syms")
    else:
        symbols_input = st.text_input("股票代码列表 (逗号分隔)", value="AAPL,MSFT,SPY", key="t3_syms_custom")

    btn3 = st.button("🔍 开始扫描", type="primary", key="t3_btn")

    if btn3:
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("请至少输入一只股票代码")
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=400)

            results = []
            progress = st.progress(0)
            for i, sym in enumerate(symbols):
                progress.progress((i + 1) / len(symbols), text=f"扫描 {sym}...")
                d = fetch_data(sym, "1d", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                if d.empty or len(d) < 250:
                    continue
                df_full = compute_all_signals(d.copy())
                df_full = detect_all_advanced_patterns(df_full)
                df_full = detect_extra_patterns(df_full)
                df_full = compute_new_indicators(df_full)
                last = df_full.iloc[-1]

                triggered = []
                for sig_col in SIGNAL_STATS.keys():
                    if sig_col in df_full.columns:
                        val = last.get(sig_col)
                        if isinstance(val, (bool, np.bool_)) and bool(val):
                            stat = SIGNAL_STATS[sig_col]
                            triggered.append({
                                "label_zh": zh_label(sig_col),
                                "alpha": stat["alpha_10d"],
                                "oos_significant": stat.get("oos_significant", False),
                            })
                triggered.sort(key=lambda t: abs(t["alpha"]), reverse=True)

                results.append({
                    "代码": sym,
                    "价格": f"${last['close']:.2f}",
                    "MACD": "↑" if last.get("macd_state") == 1 else "↓" if last.get("macd_state") == -1 else "—",
                    "200日均线之上": "是" if last.get("above_sma200") else "否",
                    "ADX": f"{last['adx']:.1f}" if pd.notna(last.get("adx")) else "—",
                    "触发的信号 (取前 5)": ", ".join(f"{t['label_zh']} ({t['alpha']:+.2f}% α)" for t in triggered[:5]) if triggered else "无",
                    "n_signals": len(triggered),
                })
            progress.empty()

            if results:
                st.markdown(f"### 扫描结果 ({len(results)} 只股票)")
                df_res = pd.DataFrame(results)
                if "n_signals" in df_res.columns:
                    df_res = df_res.drop(columns=["n_signals"])
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                st.info(
                    "**结果说明**: '触发的信号' 列每只股票最多显示 5 个信号, "
                    "按绝对 alpha 排序 (历史影响最大的排前)。"
                    "Alpha 是该信号触发后 21 年的平均 10 日收益率相对基准的差异。"
                    "**这是描述性测量, 不是买卖建议**。"
                )


# ============== Footer ==============
st.markdown("---")
st.caption(
    "**指标参考文献**: MACD (Appel 1979), KDJ/Stochastic (Lane 1950s), DMI/ADX (Wilder 1978), "
    "PSAR (Wilder 1978), Bollinger Bands (Bollinger 2001), Candlestick patterns (Nison 1991), "
    "Pivot Points (floor trader pivots). Chart patterns: Edwards & Magee, *Technical Analysis of Stock Trends* (1948). "
    "**回测研究**: 21 年日线数据, 35 只美股大盘股 (约 182,000 个观测)。"
    "已控制 look-ahead bias。摩擦未建模。"
    "**样本外验证**: 训练集 2005-2019 vs 测试集 2020-2026, 平均漂移 2.15%。"
    "过去频率不能预测未来。"
)
