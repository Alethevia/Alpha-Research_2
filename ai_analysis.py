"""
OpenAI AI 技术面解读
=====================
环境变量: OPENAI_API_KEY
模型: gpt-4o-mini (低成本, 中文响应足够好)

UI 流程:
  1. 用户点击 "🤖 AI 解读" 按钮
  2. 后端拼装 prompt: 价格 + 共振 + 各维度 + 关键指标 + 信号 + 背离 + Kronos (若有)
  3. OpenAI ChatCompletions API → 中文技术面分析
  4. 显示在 st.chat_message 中
  5. 强制免责声明结尾

设计原则:
  - 系统提示禁止 "建议买入/卖出"
  - 禁止承诺价格走势
  - 强制中文回答
  - 单次调用, 不支持多轮 (避免 token 失控)
  - 任何错误返回 {"ok": False, "error": ...} 不让 UI 崩溃
"""

from __future__ import annotations

import os
from typing import Any


SYSTEM_PROMPT = (
    "你是一个客观的金融技术分析助手。用中文回答。"
    "只描述技术面现状, 不给买卖建议。"
    "禁止使用以下措辞: 建议买入、建议卖出、推荐买、推荐卖、应该买、应该卖、值得买入。"
    "禁止预测具体价格点位。允许使用: 当前显示、技术面呈现、指标显示、共振方向、"
    "可关注、值得留意 (用于描述事实而非建议)。"
    "结构化输出: 1) 趋势综述 2) 关键指标解读 3) 信号亮点 4) 风险提示。"
    "总长度不超过 500 字。"
)


def get_api_key() -> str | None:
    """读取 OPENAI_API_KEY (env 优先, 然后 Streamlit secrets)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        secret = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


def is_available() -> bool:
    """是否能调用 OpenAI (key + 包安装均需就绪)."""
    if not get_api_key():
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def get_unavailable_reason() -> str:
    if not get_api_key():
        return "未配置 OPENAI_API_KEY 环境变量"
    try:
        import openai  # noqa: F401
        return ""
    except ImportError:
        return "未安装 openai 库 (pip install openai)"


def _format_pct(x):
    return f"{x:+.2f}%" if isinstance(x, (int, float)) else "N/A"


def _format_num(x, fmt=".2f"):
    if isinstance(x, (int, float)):
        return f"{x:{fmt}}"
    return "N/A"


def _format_bool(x):
    return "是" if x else "否"


def build_prompt(context: dict) -> str:
    """
    将技术分析上下文格式化成 user prompt。

    context 期望键:
      symbol, price, change_pct,
      confluence: {"score": ..., "label": ..., "dimensions": {...}},
      indicators: {rsi, cci, williams_r, atr, vwap,
                   supertrend_bullish, ichimoku_above_cloud, nx_bull_zone, nx_bear_zone, ...},
      signals: list[str],       # 当前触发的中文信号名
      divergences: list[str],   # 检测到的背离
      kronos: dict | None,      # Kronos/Chronos 输出 (可选)
      conformal: dict | None,   # AR(5) + conformal 输出 (可选)
    """
    sym = context.get("symbol", "未知代码")
    price = context.get("price")
    change = context.get("change_pct")

    lines = [
        f"## 当前标的: {sym}",
        f"- 收盘价: {_format_num(price)} (日内变动: {_format_pct(change)})",
        "",
    ]

    conf = context.get("confluence") or {}
    if conf:
        lines.append(f"## 共振评分: {conf.get('score', 0):.0f} / 100 → {conf.get('label', '—')}")
        dims = conf.get("dimensions") or {}
        dim_zh = {
            "trend": "趋势", "momentum": "动量",
            "overbought_oversold": "超买超卖", "volume": "成交量",
            "volatility": "波动率", "pattern": "形态",
        }
        for k, v in dims.items():
            zh = dim_zh.get(k, k)
            sc = v.get("score", 0)
            b = v.get("n_bull", 0)
            be = v.get("n_bear", 0)
            n = v.get("n_neutral", 0)
            lines.append(f"  - {zh}: 分值 {sc:+.2f} (多 {b} / 空 {be} / 中 {n})")
        lines.append("")

    ind = context.get("indicators") or {}
    if ind:
        lines.append("## 关键指标读数")
        rsi = ind.get("rsi")
        cci = ind.get("cci")
        wr = ind.get("williams_r")
        atr = ind.get("atr")
        vwap = ind.get("vwap")
        cmf = ind.get("cmf")
        lines.append(f"  - RSI(14): {_format_num(rsi, '.1f')}")
        lines.append(f"  - CCI(20): {_format_num(cci, '.0f')}")
        lines.append(f"  - Williams %R: {_format_num(wr, '.1f')}")
        lines.append(f"  - ATR(14): {_format_num(atr)}")
        lines.append(f"  - VWAP(20): {_format_num(vwap)}")
        if cmf is not None:
            lines.append(f"  - CMF(20): {_format_num(cmf, '.3f')}")

        lines.append("  - Supertrend: " +
                     ("多头" if ind.get("supertrend_bullish") else
                      "空头" if ind.get("supertrend_bearish") else "中性"))
        lines.append("  - Ichimoku 云: " +
                     ("云上 (强势)" if ind.get("ichimoku_above_cloud") else
                      "云下 (弱势)" if ind.get("ichimoku_below_cloud") else "云内 (盘整)"))
        lines.append("  - NX 通道: " +
                     ("多头排列" if ind.get("nx_bull_zone") else
                      "空头排列" if ind.get("nx_bear_zone") else "中性"))
        lines.append("  - VWAP 关系: " +
                     ("价格在 VWAP 上方" if ind.get("above_vwap") else
                      "价格在 VWAP 下方" if ind.get("below_vwap") else "—"))
        lines.append("  - 200 日均线: " + _format_bool(ind.get("above_sma200")) + "上方")
        if ind.get("squeeze_on"):
            lines.append("  - TTM Squeeze: 挤压中 (低波动, 等待突破)")
        if ind.get("squeeze_fired_up"):
            lines.append("  - TTM Squeeze: 向上释放 ⚡")
        if ind.get("squeeze_fired_dn"):
            lines.append("  - TTM Squeeze: 向下释放 ⚡")
        lines.append("")

    sigs = context.get("signals") or []
    if sigs:
        lines.append(f"## 当前触发的技术信号 ({len(sigs)} 个)")
        for s in sigs[:25]:  # 限制 25 个避免 token 爆炸
            lines.append(f"  - {s}")
        if len(sigs) > 25:
            lines.append(f"  - ...及另外 {len(sigs) - 25} 个")
        lines.append("")
    else:
        lines.append("## 当前触发的技术信号: 无")
        lines.append("")

    divs = context.get("divergences") or []
    if divs:
        lines.append("## 检测到的背离")
        for d in divs:
            lines.append(f"  - {d}")
        lines.append("")

    conformal = context.get("conformal")
    if conformal and conformal.get("ok"):
        lines.append("## AR(5) + Conformal 概率预测")
        lines.append(f"  - {conformal.get('horizon')}d 点预测: {conformal.get('point_pct'):+.2f}%")
        lines.append(f"  - {int((1 - conformal.get('alpha', 0.1)) * 100)}% 置信区间: "
                     f"[{conformal.get('lower_pct'):+.1f}%, {conformal.get('upper_pct'):+.1f}%]")
        lines.append(f"  - P(上涨): {conformal.get('p_up', 0) * 100:.1f}%")
        lines.append("")

    kronos = context.get("kronos")
    if kronos and kronos.get("ok"):
        lines.append("## Chronos AI 预测")
        lines.append(f"  - 中位数 {kronos.get('horizon')}d 收益: {kronos.get('point_pct'):+.2f}%")
        p_up = kronos.get("p_up_native")
        if p_up is not None:
            lines.append(f"  - P(上涨): {p_up * 100:.1f}%")
        lines.append("")

    lines.append("---")
    lines.append("请基于以上数据, 用中文给出 4 部分技术面解读:")
    lines.append("1. 趋势综述 (短中长期方向)")
    lines.append("2. 关键指标解读 (哪些信号支持趋势, 哪些矛盾)")
    lines.append("3. 信号亮点 (触发的最具描述价值的信号)")
    lines.append("4. 风险提示 (技术面潜在弱点)")
    lines.append("")
    lines.append("严格禁止: 建议买入/卖出, 推荐买/卖, 应该买/卖, 预测具体价格点位。")

    return "\n".join(lines)


def call_openai(prompt: str,
                model: str = "gpt-4o-mini",
                temperature: float = 0.3,
                max_tokens: int = 1200,
                timeout: int = 45) -> dict:
    """
    调用 OpenAI ChatCompletions API. 错误模式 →
    {"ok": False, "error": str, "answer": None}
    成功:
    {"ok": True, "error": None, "answer": str,
     "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}}
    """
    key = get_api_key()
    if not key:
        return {"ok": False, "error": "未配置 OPENAI_API_KEY 环境变量", "answer": None}

    try:
        from openai import OpenAI
        from openai import APIError, APIConnectionError, RateLimitError, AuthenticationError
    except ImportError as e:
        return {"ok": False, "error": f"openai 包未安装: {e}", "answer": None}

    try:
        client = OpenAI(api_key=key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not resp.choices:
            return {"ok": False, "error": "OpenAI 返回为空", "answer": None}
        answer = resp.choices[0].message.content or ""

        usage_obj = getattr(resp, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }

        return {
            "ok": True,
            "error": None,
            "answer": answer.strip(),
            "usage": usage,
            "model": getattr(resp, "model", model),
        }
    except AuthenticationError as e:
        return {"ok": False, "error": f"OpenAI 鉴权失败 (key 无效): {e}", "answer": None}
    except RateLimitError as e:
        return {"ok": False, "error": f"OpenAI 限流: {e}", "answer": None}
    except APIConnectionError as e:
        return {"ok": False, "error": f"连接 OpenAI 失败: {e}", "answer": None}
    except APIError as e:
        return {"ok": False, "error": f"OpenAI API 错误: {e}", "answer": None}
    except Exception as e:
        return {"ok": False, "error": f"未预期错误: {type(e).__name__}: {e}", "answer": None}


def analyze(context: dict, model: str = "gpt-4o-mini") -> dict:
    """便捷入口: build_prompt + call_openai 一气呵成."""
    prompt = build_prompt(context)
    return call_openai(prompt, model=model)
