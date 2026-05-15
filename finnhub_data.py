"""
Finnhub.io 免费版数据接入
Finnhub Free-tier Data Client
================================
免费层提供:
  - 公司新闻 (company-news)
  - 新闻情绪 (news-sentiment)
  - 内幕交易 (stock/insider-transactions)
  - 分析师推荐 (stock/recommendation)
  - 财报历史/惊喜 (calendar/earnings, stock/earnings)
  - 60 次/分钟 速率限制 (足够本工具单股查询使用)

凭证:
  从 https://finnhub.io 注册免费账户, 拷贝 API key, 设置环境变量
  FINNHUB_API_KEY=<your-key>。也可通过 Streamlit secrets 注入。

设计:
  - 只用 urllib (无新依赖, 避免版本冲突)
  - 所有函数都返回 dict: {"ok": bool, "data": ..., "error": str|None}
  - 网络/JSON 错误静默捕获 → ok=False (调用方判断)
  - 用 functools.lru_cache 做进程内缓存 (单 TTL 由调用方控制)

法律:
  分析师推荐 (买/持/卖) 是 *第三方机构观点*, 不构成本工具的建议。
  UI 层必须明确标注 "外部数据" 并禁止任何 "我们推荐..." 措辞。
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone


_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 10


def get_api_key() -> str | None:
    """读取 API key。优先环境变量, 其次 Streamlit secrets (如可用)。"""
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if key:
        return key
    # Streamlit secrets (只有在 Streamlit 上下文中才可用)
    try:
        import streamlit as st
        secret = st.secrets.get("FINNHUB_API_KEY") if hasattr(st, "secrets") else None
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


def _call(path: str, params: dict) -> dict:
    key = get_api_key()
    if not key:
        return {"ok": False, "data": None, "error": "no_key"}

    qs = urllib.parse.urlencode({**params, "token": key})
    url = f"{_BASE}/{path.lstrip('/')}?{qs}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "AlphaResearch-Educational/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            body = r.read()
            return {"ok": True, "data": json.loads(body), "error": None}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="ignore")[:300]
        except Exception:
            err_body = ""
        return {"ok": False, "data": None,
                "error": f"HTTP {e.code}: {err_body}"}
    except urllib.error.URLError as e:
        return {"ok": False, "data": None, "error": f"network: {e.reason}"}
    except (json.JSONDecodeError, ValueError) as e:
        return {"ok": False, "data": None, "error": f"json: {e}"}
    except Exception as e:
        return {"ok": False, "data": None, "error": f"unexpected: {e}"}


# =============================================================================
# 公司新闻 (Company News)
# =============================================================================
def get_company_news(symbol: str, days_back: int = 7, limit: int = 20) -> dict:
    """
    获取一只股票最近 N 天的公司新闻。
    Returns: {"ok": bool, "data": list[{datetime, headline, source, summary, url, image, category}], "error"}
    """
    if not symbol:
        return {"ok": False, "data": [], "error": "no_symbol"}

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=max(1, days_back))
    r = _call("company-news", {
        "symbol": symbol.upper(),
        "from": start.strftime("%Y-%m-%d"),
        "to": today.strftime("%Y-%m-%d"),
    })
    if not r["ok"]:
        return {"ok": False, "data": [], "error": r["error"]}

    items = r["data"] or []
    # 时间戳 -> ISO 字符串, 截取前 N 条
    cleaned = []
    for it in items[:limit]:
        ts = it.get("datetime")
        if isinstance(ts, (int, float)) and ts > 0:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            ts_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        else:
            ts_str = ""
        cleaned.append({
            "datetime": ts_str,
            "headline": it.get("headline") or "",
            "source": it.get("source") or "",
            "summary": (it.get("summary") or "")[:400],
            "url": it.get("url") or "",
            "category": it.get("category") or "",
            "image": it.get("image") or "",
        })
    return {"ok": True, "data": cleaned, "error": None}


# =============================================================================
# 新闻情绪 (News Sentiment)
# =============================================================================
def get_news_sentiment(symbol: str) -> dict:
    """
    Finnhub 的"新闻情绪"分数 (基于一周新闻).
    Returns 字段:
      buzz: {articlesInLastWeek, buzz, weeklyAverage}
      companyNewsScore (0~1): 越接近 1 越正面
      sectorAverageBullishPercent
      sentiment: {bearishPercent, bullishPercent}
    """
    if not symbol:
        return {"ok": False, "data": None, "error": "no_symbol"}
    r = _call("news-sentiment", {"symbol": symbol.upper()})
    if not r["ok"]:
        return r
    d = r["data"] or {}
    # 归一化输出
    return {
        "ok": True,
        "error": None,
        "data": {
            "company_news_score": d.get("companyNewsScore"),
            "sector_avg_bullish_pct": d.get("sectorAverageBullishPercent"),
            "sector_avg_news_score": d.get("sectorAverageNewsScore"),
            "articles_last_week": (d.get("buzz") or {}).get("articlesInLastWeek"),
            "buzz": (d.get("buzz") or {}).get("buzz"),
            "weekly_avg_articles": (d.get("buzz") or {}).get("weeklyAverage"),
            "bullish_pct": (d.get("sentiment") or {}).get("bullishPercent"),
            "bearish_pct": (d.get("sentiment") or {}).get("bearishPercent"),
        },
    }


# =============================================================================
# 内幕交易 (Insider Transactions)
# =============================================================================
def get_insider_transactions(symbol: str, limit: int = 20) -> dict:
    """
    最近内幕交易 (高管/董事的买入卖出申报).
    Returns: list[{name, share, transactionDate, transactionPrice, transactionCode}]
      transactionCode: P=Purchase, S=Sale, A=Award, M=Option exercise, ...
    """
    if not symbol:
        return {"ok": False, "data": [], "error": "no_symbol"}
    r = _call("stock/insider-transactions", {"symbol": symbol.upper()})
    if not r["ok"]:
        return {"ok": False, "data": [], "error": r["error"]}
    items = (r["data"] or {}).get("data") or []
    items = sorted(items, key=lambda x: x.get("transactionDate") or "", reverse=True)
    cleaned = []
    for it in items[:limit]:
        cleaned.append({
            "name": it.get("name") or "",
            "share": it.get("share"),  # 净变动股数 (可负)
            "change": it.get("change"),  # 变动股数 (有符号)
            "transactionDate": it.get("transactionDate") or "",
            "transactionPrice": it.get("transactionPrice"),
            "transactionCode": it.get("transactionCode") or "",
            "filingDate": it.get("filingDate") or "",
        })
    return {"ok": True, "data": cleaned, "error": None}


# =============================================================================
# 分析师推荐汇总 (Recommendation Trends)
# =============================================================================
def get_recommendations(symbol: str) -> dict:
    """
    分析师推荐汇总 (过去 4 个月每月一行). **外部第三方观点**, 不是本工具建议。
    Returns: list[{period, buy, hold, sell, strongBuy, strongSell}]
    """
    if not symbol:
        return {"ok": False, "data": [], "error": "no_symbol"}
    r = _call("stock/recommendation", {"symbol": symbol.upper()})
    if not r["ok"]:
        return {"ok": False, "data": [], "error": r["error"]}
    items = r["data"] or []
    items = sorted(items, key=lambda x: x.get("period") or "", reverse=True)
    cleaned = []
    for it in items[:6]:
        cleaned.append({
            "period": it.get("period") or "",
            "strongBuy": it.get("strongBuy", 0),
            "buy": it.get("buy", 0),
            "hold": it.get("hold", 0),
            "sell": it.get("sell", 0),
            "strongSell": it.get("strongSell", 0),
        })
    return {"ok": True, "data": cleaned, "error": None}


# =============================================================================
# 财报历史 / 惊喜 (Earnings Surprise)
# =============================================================================
def get_earnings_surprise(symbol: str) -> dict:
    """
    最近 4 季度财报 vs 分析师预期.
    Returns: list[{period, actual, estimate, surprise, surprisePercent}]
    """
    if not symbol:
        return {"ok": False, "data": [], "error": "no_symbol"}
    r = _call("stock/earnings", {"symbol": symbol.upper(), "limit": 4})
    if not r["ok"]:
        return {"ok": False, "data": [], "error": r["error"]}
    items = r["data"] or []
    items = sorted(items, key=lambda x: x.get("period") or "", reverse=True)
    cleaned = []
    for it in items[:4]:
        cleaned.append({
            "period": it.get("period") or "",
            "actual": it.get("actual"),
            "estimate": it.get("estimate"),
            "surprise": it.get("surprise"),
            "surprisePercent": it.get("surprisePercent"),
        })
    return {"ok": True, "data": cleaned, "error": None}
