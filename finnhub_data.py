"""
Finnhub.io 完整整合 (使用官方 finnhub-python 客户端)
====================================================
环境变量: FINNHUB_API_KEY (从 https://finnhub.io 免费获取, 60 次/分钟)

5 个数据源 (全部为真实 API 调用, 无 mock):
  1. company_news               — 公司新闻 + 头条情绪 (按 Finnhub 头条分类聚合估计)
  2. stock_insider_transactions — 高管/董事 SEC Form 4 申报
  3. recommendation_trends      — 分析师评级汇总 (强买/买/持/卖/强卖)
  4. earnings_calendar          — 财报日历 + 实际 vs 预期 EPS
  5. company_basic_financials   — 基础财务指标 (P/E, ROE, 营收增长等)

每个函数返回:
  {"ok": bool, "data": <list|dict>, "error": str|None, "raw": <原始 JSON>}

错误模式 (全部静默捕获, 不让 Streamlit 崩溃):
  - 无 key → ok=False, error="no_key"
  - 网络错误 → ok=False, error="network: ..."
  - API 错误 (rate limit/403/404) → ok=False, error="api: ..."
  - 空响应 → ok=True, data=[], error=None

法律提示:
  分析师评级是第三方机构观点, 不是本工具的建议。
  情绪分基于头条 NLP, 可能误读上下文。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any


_CLIENT_CACHE: dict[str, Any] = {}


def get_api_key() -> str | None:
    """读取 API key (env 优先, 然后 Streamlit secrets)."""
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        secret = st.secrets.get("FINNHUB_API_KEY") if hasattr(st, "secrets") else None
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


def _client():
    """缓存的 finnhub.Client. 没 key 返回 None."""
    key = get_api_key()
    if not key:
        return None
    if _CLIENT_CACHE.get("key") == key and _CLIENT_CACHE.get("client") is not None:
        return _CLIENT_CACHE["client"]
    try:
        import finnhub
        client = finnhub.Client(api_key=key)
        _CLIENT_CACHE["key"] = key
        _CLIENT_CACHE["client"] = client
        return client
    except ImportError:
        return None
    except Exception:
        return None


def _err(msg: str, data=None):
    return {"ok": False, "data": data if data is not None else [], "error": msg, "raw": None}


def _ok(data, raw=None):
    return {"ok": True, "data": data, "error": None, "raw": raw}


# Heuristic English-only sentiment lexicon for headline scoring.
# Finnhub's free tier does NOT include per-article sentiment in company_news;
# we derive a lightweight score from headlines so the UI can show a column.
_POS_WORDS = (
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgrades", "upgraded", "raises", "raised", "outperform",
    "record", "strong", "strength", "jumps", "jumped", "bullish", "buy",
    "wins", "won", "growth", "boost", "boosts", "boosted", "rises", "rose",
    "expand", "expands", "expanded", "innovation", "approved", "approves",
)
_NEG_WORDS = (
    "miss", "misses", "missed", "plunge", "plunges", "slump", "slumps",
    "downgrade", "downgrades", "downgraded", "cut", "cuts", "lowered", "lowers",
    "underperform", "weak", "weakness", "drop", "drops", "dropped",
    "bearish", "sell", "loss", "losses", "warns", "warning", "fall", "fell",
    "fraud", "lawsuit", "investigation", "probe", "decline", "declines",
    "shrink", "shrinks", "shrank", "concern", "concerns", "risk",
)


def _score_headline(text: str) -> float:
    """
    -1..+1 启发式情绪分 (基于英文头条词频).
    Finnhub 免费版的 company_news 返回中不包含 article-level sentiment,
    所以我们在这里做一个透明的 fallback. UI 必须标注 "启发式".
    """
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in t)
    neg = sum(1 for w in _NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, pos + neg)


# =============================================================================
# 1. Company News + 情绪
# =============================================================================
def get_company_news(symbol: str, days_back: int = 7, limit: int = 25) -> dict:
    """
    finnhub.Client.company_news(symbol, _from, to) → list[dict].
    每条新闻包含: category, datetime (unix), headline, id, image, related,
                  source, summary, url.
    """
    if not symbol:
        return _err("no_symbol")
    client = _client()
    if client is None:
        return _err("no_key")

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=max(1, days_back))
    try:
        raw = client.company_news(
            symbol.upper(),
            _from=start.strftime("%Y-%m-%d"),
            to=today.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        return _err(f"api: {e}")

    if not isinstance(raw, list):
        return _ok([], raw=raw)

    items = []
    for n in raw[:limit]:
        if not isinstance(n, dict):
            continue
        ts = n.get("datetime")
        if isinstance(ts, (int, float)) and ts > 0:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            sort_key = float(ts)
        else:
            dt_str = ""
            sort_key = 0.0
        headline = (n.get("headline") or "").strip()
        score = _score_headline(headline + " " + (n.get("summary") or ""))
        items.append({
            "datetime": dt_str,
            "_sort": sort_key,
            "headline": headline,
            "source": n.get("source") or "",
            "summary": (n.get("summary") or "")[:500],
            "url": n.get("url") or "",
            "category": n.get("category") or "",
            "image": n.get("image") or "",
            "sentiment_score": round(score, 3),
            "sentiment_label": ("positive" if score > 0.2 else
                                "negative" if score < -0.2 else "neutral"),
        })
    items.sort(key=lambda x: x["_sort"], reverse=True)
    for it in items:
        it.pop("_sort", None)

    # 汇总指标
    if items:
        avg_score = sum(x["sentiment_score"] for x in items) / len(items)
        pos = sum(1 for x in items if x["sentiment_label"] == "positive")
        neg = sum(1 for x in items if x["sentiment_label"] == "negative")
    else:
        avg_score, pos, neg = 0.0, 0, 0

    return _ok({
        "articles": items,
        "summary": {
            "total": len(items),
            "avg_headline_score": round(avg_score, 3),
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": len(items) - pos - neg,
        },
    }, raw=raw)


# =============================================================================
# 2. Insider transactions (SEC Form 4)
# =============================================================================
def get_insider_transactions(symbol: str, days_back: int = 180, limit: int = 30) -> dict:
    if not symbol:
        return _err("no_symbol")
    client = _client()
    if client is None:
        return _err("no_key")

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    try:
        raw = client.stock_insider_transactions(
            symbol.upper(),
            _from=start.strftime("%Y-%m-%d"),
            to=today.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        return _err(f"api: {e}")

    data_list = []
    if isinstance(raw, dict):
        data_list = raw.get("data") or []
    elif isinstance(raw, list):
        data_list = raw

    items = []
    for it in data_list:
        if not isinstance(it, dict):
            continue
        items.append({
            "name": (it.get("name") or "").strip(),
            "share": it.get("share"),
            "change": it.get("change"),
            "filingDate": it.get("filingDate") or "",
            "transactionDate": it.get("transactionDate") or "",
            "transactionPrice": it.get("transactionPrice"),
            "transactionCode": (it.get("transactionCode") or "").strip(),
        })
    items.sort(key=lambda x: x["transactionDate"], reverse=True)
    items = items[:limit]

    # 汇总
    buy_shares = sum(int(x["change"]) for x in items
                     if isinstance(x["change"], (int, float)) and x["change"] > 0)
    sell_shares = -sum(int(x["change"]) for x in items
                       if isinstance(x["change"], (int, float)) and x["change"] < 0)
    return _ok({
        "transactions": items,
        "summary": {
            "n": len(items),
            "buy_shares": buy_shares,
            "sell_shares": sell_shares,
            "net_shares": buy_shares - sell_shares,
        },
    }, raw=raw)


# =============================================================================
# 3. Analyst recommendation trends
# =============================================================================
def get_recommendations(symbol: str, limit: int = 6) -> dict:
    if not symbol:
        return _err("no_symbol")
    client = _client()
    if client is None:
        return _err("no_key")
    try:
        raw = client.recommendation_trends(symbol.upper())
    except Exception as e:
        return _err(f"api: {e}")

    if not isinstance(raw, list):
        return _ok([], raw=raw)

    rows = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        rows.append({
            "period": r.get("period") or "",
            "strongBuy": int(r.get("strongBuy") or 0),
            "buy": int(r.get("buy") or 0),
            "hold": int(r.get("hold") or 0),
            "sell": int(r.get("sell") or 0),
            "strongSell": int(r.get("strongSell") or 0),
        })
    rows.sort(key=lambda x: x["period"], reverse=True)
    rows = rows[:limit]
    return _ok(rows, raw=raw)


# =============================================================================
# 4. Earnings calendar (历史 + 即将到来)
# =============================================================================
def get_earnings_calendar(symbol: str, days_back: int = 365, days_ahead: int = 90) -> dict:
    if not symbol:
        return _err("no_symbol")
    client = _client()
    if client is None:
        return _err("no_key")

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    end = today + timedelta(days=days_ahead)
    try:
        raw = client.earnings_calendar(
            _from=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
            symbol=symbol.upper(),
            international=False,
        )
    except Exception as e:
        return _err(f"api: {e}")

    cal_list = []
    if isinstance(raw, dict):
        cal_list = raw.get("earningsCalendar") or []
    elif isinstance(raw, list):
        cal_list = raw

    upcoming = []
    historical = []
    for it in cal_list:
        if not isinstance(it, dict):
            continue
        item = {
            "date": it.get("date") or "",
            "hour": it.get("hour") or "",
            "epsActual": it.get("epsActual"),
            "epsEstimate": it.get("epsEstimate"),
            "revenueActual": it.get("revenueActual"),
            "revenueEstimate": it.get("revenueEstimate"),
            "quarter": it.get("quarter"),
            "year": it.get("year"),
            "symbol": it.get("symbol") or symbol.upper(),
        }
        eps_act = item.get("epsActual")
        eps_est = item.get("epsEstimate")
        if isinstance(eps_act, (int, float)) and isinstance(eps_est, (int, float)) and eps_est != 0:
            item["surprise_pct"] = round((eps_act - eps_est) / abs(eps_est) * 100, 2)
        else:
            item["surprise_pct"] = None

        d = item["date"]
        if d and d >= today.strftime("%Y-%m-%d"):
            upcoming.append(item)
        else:
            historical.append(item)

    historical.sort(key=lambda x: x["date"], reverse=True)
    upcoming.sort(key=lambda x: x["date"])
    return _ok({
        "upcoming": upcoming[:4],
        "historical": historical[:8],
    }, raw=raw)


# =============================================================================
# 5. Basic financials
# =============================================================================
def get_basic_financials(symbol: str) -> dict:
    if not symbol:
        return _err("no_symbol")
    client = _client()
    if client is None:
        return _err("no_key")
    try:
        raw = client.company_basic_financials(symbol.upper(), "all")
    except Exception as e:
        return _err(f"api: {e}")

    metric = (raw or {}).get("metric") if isinstance(raw, dict) else {}
    metric = metric or {}

    def _g(k):
        v = metric.get(k)
        return v if isinstance(v, (int, float)) else None

    parsed = {
        "pe_ttm": _g("peTTM"),
        "pe_basic_exclExtraTTM": _g("peBasicExclExtraTTM"),
        "pb": _g("pbAnnual") or _g("pbQuarterly"),
        "ps_ttm": _g("psTTM"),
        "ev_to_ebitda_ttm": _g("currentEv/freeCashFlowTTM"),
        "dividend_yield_ttm": _g("dividendYieldIndicatedAnnual"),
        "roe_ttm": _g("roeTTM"),
        "roa_ttm": _g("roaTTM"),
        "gross_margin_ttm": _g("grossMarginTTM"),
        "net_margin_ttm": _g("netProfitMarginTTM"),
        "revenue_growth_3y": _g("revenueGrowth3Y"),
        "revenue_growth_5y": _g("revenueGrowth5Y"),
        "eps_growth_3y": _g("epsGrowth3Y"),
        "eps_growth_5y": _g("epsGrowth5Y"),
        "current_ratio": _g("currentRatioAnnual"),
        "debt_to_equity": _g("totalDebt/totalEquityAnnual"),
        "beta": _g("beta"),
        "52_week_high": _g("52WeekHigh"),
        "52_week_low": _g("52WeekLow"),
    }
    return _ok(parsed, raw=raw)
