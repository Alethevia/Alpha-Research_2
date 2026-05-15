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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

from extended_signals import compute_all_signals
from advanced_patterns import detect_all_advanced_patterns
from extra_patterns import detect_extra_patterns
from historical_stats import SIGNAL_STATS, BASELINE_10D, format_freq_text, stability_badge
from new_indicators import compute_new_indicators
from confluence import compute_confluence_summary

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
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol_input = st.text_input("股票代码", value="AAPL", key="t1_sym",
                                      placeholder="例如: AAPL, TSLA, SPY, BABA")
    with col2:
        period = st.selectbox("时间窗口", ["3 个月", "6 个月", "1 年", "2 年"], index=2, key="t1_period")
    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("📥 加载数据", type="primary", use_container_width=True, key="t1_btn")

    days = {"3 个月": 90, "6 个月": 180, "1 年": 365, "2 年": 730}[period]
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

                # === 图表 ===
                st.markdown("## 📈 价格图表")
                fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1, 1, 1]})
                ax = axes[0]
                ax.plot(disp["timestamp"], disp["close"], color="black", lw=1.2, label="Close")
                if "sma200" in disp.columns:
                    ax.plot(disp["timestamp"], disp["sma200"], color="purple", lw=1, alpha=0.7, label="SMA200")
                if "sma50" in disp.columns:
                    ax.plot(disp["timestamp"], disp["sma50"], color="orange", lw=0.8, alpha=0.7, label="SMA50")
                gc_dates = disp[disp["macd_gc"] == True] if "macd_gc" in disp.columns else disp.iloc[0:0]
                dc_dates = disp[disp["macd_dc"] == True] if "macd_dc" in disp.columns else disp.iloc[0:0]
                if not gc_dates.empty:
                    ax.scatter(gc_dates["timestamp"], gc_dates["close"], color="green", s=80, marker="^", zorder=5, label="MACD GC", edgecolors="darkgreen")
                if not dc_dates.empty:
                    ax.scatter(dc_dates["timestamp"], dc_dates["close"], color="red", s=80, marker="v", zorder=5, label="MACD DC", edgecolors="darkred")
                db_dates = disp[disp["pat_double_bottom"] == True] if "pat_double_bottom" in disp.columns else disp.iloc[0:0]
                dt_dates = disp[disp["pat_double_top"] == True] if "pat_double_top" in disp.columns else disp.iloc[0:0]
                if not db_dates.empty:
                    ax.scatter(db_dates["timestamp"], db_dates["close"], color="blue", s=120, marker="D", zorder=5, label="Double Bottom", edgecolors="darkblue")
                if not dt_dates.empty:
                    ax.scatter(dt_dates["timestamp"], dt_dates["close"], color="darkred", s=120, marker="X", zorder=5, label="Double Top", edgecolors="black")
                ax.set_ylabel("Price ($)")
                ax.set_title(f"{sym} - Price + SMAs + Pattern Markers")
                ax.legend(loc="upper left", fontsize=7)
                ax.grid(alpha=0.3)

                ax = axes[1]
                if "macd_hist" in disp.columns:
                    colors = ["g" if pd.notna(h) and h > 0 else ("r" if pd.notna(h) else "lightgray")
                              for h in disp["macd_hist"]]
                    ax.bar(disp["timestamp"], disp["macd_hist"], color=colors, alpha=0.6, width=0.7)
                    ax.axhline(0, color="black", lw=0.5)
                ax.set_ylabel("MACD Hist")
                ax.grid(alpha=0.3)

                ax = axes[2]
                if "kdj_k" in disp.columns:
                    ax.plot(disp["timestamp"], disp["kdj_k"], color="blue", lw=1, label="K")
                    ax.plot(disp["timestamp"], disp["kdj_d"], color="orange", lw=1, label="D")
                    ax.axhline(80, color="red", ls=":", alpha=0.4)
                    ax.axhline(20, color="green", ls=":", alpha=0.4)
                ax.set_ylabel("KDJ"); ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(alpha=0.3)

                ax = axes[3]
                if "adx" in disp.columns:
                    ax.plot(disp["timestamp"], disp["adx"], color="purple", lw=1.2, label="ADX")
                    ax.plot(disp["timestamp"], disp["plus_di"], color="green", lw=0.8, label="+DI")
                    ax.plot(disp["timestamp"], disp["minus_di"], color="red", lw=0.8, label="-DI")
                    ax.axhline(25, color="gray", ls=":", alpha=0.4)
                ax.set_ylabel("ADX/DMI"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

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
