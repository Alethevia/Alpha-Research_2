"""
Kronos / Chronos AI 概率预测包装
================================
默认使用 Amazon Chronos-Bolt-Tiny (~9M 参数, Apache 2.0, CPU 可跑)。
也可切换到 NeoQuasar/Kronos-mini (~4M, MIT) 通过 model_name 参数。

设计:
  - lazy import: torch + chronos 都是大依赖, 只在首次调用时导入
  - 失败模式: 如果包未安装/模型下载失败, 返回 {"ok": False, "error": "..."}
  - 输出归一化: 所有函数返回 log return (累计) 而不是绝对价格,
    这样能直接喂给 conformal.conformalize_external_forecaster

输出:
  predict_distribution(close, horizon) → dict
    - point_log_return: 中位数 (median) log return
    - quantiles: {0.1, 0.25, 0.5, 0.75, 0.9} → log return
    - p_up_native: 原生 P(price > current) 估计 (基于全部样本)
    - p_up_5pct: 原生 P(price > current * 1.05)

部署提示:
  这些 ML 包很大 (~1GB). 如果不需要 AI 预测可以从 requirements.txt 删除
  torch + chronos-forecasting 这两行 — 应用会自动隐藏 AI 功能。
"""

from __future__ import annotations

import math
import numpy as np


_MODEL_CACHE = {}
_IMPORT_ERROR = None
_LOAD_ERROR = None  # 最近一次模型下载/加载失败的原因 (HF hub 404/403, 磁盘满, ...)


def _try_import():
    """
    懒加载 torch + chronos. 第一次失败的原因被缓存到 _IMPORT_ERROR。
    """
    global _IMPORT_ERROR
    try:
        import torch  # noqa: F401
        from chronos import BaseChronosPipeline  # noqa: F401
        _IMPORT_ERROR = None
        return True
    except ImportError as e:
        _IMPORT_ERROR = (
            f"chronos / torch 未安装 ({e}). 启用 AI 预测请安装:\n"
            f"  pip install chronos-forecasting torch --index-url https://download.pytorch.org/whl/cpu"
        )
        return False
    except Exception as e:
        _IMPORT_ERROR = f"导入失败: {e}"
        return False


def is_available() -> bool:
    """当前环境是否能运行 AI 预测."""
    return _try_import()


def get_import_error() -> str | None:
    """读取最近一次导入失败的原因 (供 UI 展示)."""
    if _IMPORT_ERROR is None:
        _try_import()
    return _IMPORT_ERROR


def _load_pipeline(model_name: str = "amazon/chronos-bolt-tiny"):
    """
    加载并缓存 Chronos pipeline。任何下载/加载错误都被捕获 → 返回 None,
    原因写到 _LOAD_ERROR 给 UI 展示。永不让上层抛异常。

    可能的失败:
      - 无网络 (HF Hub 不可达, 403/404, 超时)
      - HuggingFace 仓库不存在
      - 本地磁盘满
      - torch 不兼容 (旧版 / GPU only build)
    """
    global _LOAD_ERROR
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    if not _try_import():
        return None

    try:
        import torch
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        _MODEL_CACHE[model_name] = pipeline
        _LOAD_ERROR = None
        return pipeline
    except Exception as e:
        # 通用兜底 — HFHubHTTPError, OSError, ConnectionError, RepositoryNotFoundError 等
        _LOAD_ERROR = (
            f"模型加载失败 ({type(e).__name__}): {str(e)[:300]}\n\n"
            f"常见原因: \n"
            f"  - HuggingFace Hub 不可达 (网络/代理/区域限制)\n"
            f"  - 模型仓库 '{model_name}' 不存在或需要 token\n"
            f"  - HuggingFace Spaces 免费 CPU 实例内存不足 (~16GB), 此模型 ~40MB 通常够\n"
            f"  - torch 版本不兼容 (需要 >=2.1)"
        )
        return None


def get_load_error() -> str | None:
    """返回最近一次模型加载失败的描述, 供 UI 展示."""
    return _LOAD_ERROR


# =============================================================================
# 主入口: 预测分布
# =============================================================================
def predict_distribution(
    close: np.ndarray,
    horizon: int = 10,
    model_name: str = "amazon/chronos-bolt-tiny",
    context_length: int = 256,
    quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict:
    """
    预测未来 horizon 个交易日的累计 log return 分布。

    Parameters
    ----------
    close : 1-D array of close prices, 升序
    horizon : int, 预测期 (交易日)
    model_name : Chronos 模型名 (HF hub id)
    context_length : 用于喂模型的最近 N 根 bar (Chronos-Bolt 默认 512)
    quantile_levels : 输出分位数 levels

    Returns
    -------
    dict:
      ok, error, model, horizon
      point_log_return  : 中位数累计 log return
      point_pct         : 中位数累计 % return
      quantiles_log     : {q_level: log_return}
      quantiles_pct     : {q_level: pct_return}
      p_up_native       : 该模型估计的 P(price > current)
      p_up_5pct_native  : 该模型估计的 P(price > current * 1.05)
      p_dn_5pct_native  : 该模型估计的 P(price < current * 0.95)
      n_samples         : 使用的样本数 (Chronos 内部抽样)
    """
    if close is None or len(close) < 60:
        return {"ok": False, "error": f"need >=60 close points, got {len(close) if close is not None else 0}"}

    if not _try_import():
        return {"ok": False, "error": get_import_error()}

    import torch

    pipeline = _load_pipeline(model_name)
    if pipeline is None:
        # _LOAD_ERROR set by _load_pipeline if it tried and failed
        load_err = get_load_error()
        imp_err = get_import_error()
        return {"ok": False, "error": load_err or imp_err or "pipeline load failed"}

    # 切片到 context_length
    ctx = np.asarray(close, dtype=float)[-context_length:]
    last_close = float(ctx[-1])
    if last_close <= 0:
        return {"ok": False, "error": f"invalid last close: {last_close}"}

    try:
        # Chronos predict_quantiles accepts a 1D tensor (single series) or a list/batch
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32)
        # 注意: predict_quantiles 返回的是绝对价格预测 (不是 log return)
        quantile_preds, _mean = pipeline.predict_quantiles(
            inputs=ctx_tensor,
            prediction_length=horizon,
            quantile_levels=list(quantile_levels),
        )
        # quantile_preds shape: (1, horizon, n_quantiles) for single-series input
        qp = quantile_preds.squeeze(0).cpu().numpy()  # (horizon, n_quantiles)

        # 也获取原始样本以估计 P(up).
        # ChronosBolt 是 quantile-head 模型, 没有样本输出; 老 Chronos T5 有.
        # 任何失败都退化到分位数插值估计.
        try:
            raw = pipeline.predict(
                inputs=ctx_tensor,
                prediction_length=horizon,
            )
            arr = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
            # shape 可能是 (1, n_samples, horizon) 或 (1, horizon, n_quantiles)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            # 我们只在第二维 > n_quantiles 时认为是真正的样本
            if arr.ndim == 2 and arr.shape[0] >= 10 and arr.shape[1] == horizon:
                samples = arr  # (n_samples, horizon)
            elif arr.ndim == 2 and arr.shape[0] == horizon and arr.shape[1] >= 10:
                samples = arr.T  # 维度颠倒, 转置
            else:
                samples = None
        except Exception:
            samples = None

    except Exception as e:
        return {"ok": False, "error": f"chronos inference failed: {e}"}

    # qp[-1, :] 是第 horizon 天 (最远) 的各分位数
    final_quantiles_price = qp[-1, :]  # absolute prices
    quantiles_log = {}
    quantiles_pct = {}
    for i, q in enumerate(quantile_levels):
        price_q = float(final_quantiles_price[i])
        if price_q <= 0:
            continue
        log_ret = math.log(price_q / last_close)
        quantiles_log[q] = round(log_ret, 5)
        quantiles_pct[q] = round((math.exp(log_ret) - 1) * 100, 2)

    median_q = quantile_levels[len(quantile_levels) // 2]
    point_log = quantiles_log.get(median_q, 0.0)

    # P(up) 等概率
    p_up = p_up_5 = p_dn_5 = None
    n_samples = 0
    if samples is not None:
        # samples shape: (n_samples, horizon)
        final_prices = samples[:, -1]  # (n_samples,)
        valid = final_prices[np.isfinite(final_prices) & (final_prices > 0)]
        if len(valid) >= 10:
            p_up = float((valid > last_close).mean())
            p_up_5 = float((valid > last_close * 1.05).mean())
            p_dn_5 = float((valid < last_close * 0.95).mean())
            n_samples = int(len(valid))

    if p_up is None:
        # 用分位数线性插值估计 (近似)
        sorted_q = sorted(quantile_levels)
        sorted_prices = [float(final_quantiles_price[list(quantile_levels).index(q)])
                         for q in sorted_q]
        # 找 last_close 在哪两个分位数之间
        if last_close <= sorted_prices[0]:
            p_up = 1.0 - sorted_q[0]
        elif last_close >= sorted_prices[-1]:
            p_up = 1.0 - sorted_q[-1]
        else:
            for i in range(len(sorted_prices) - 1):
                if sorted_prices[i] <= last_close <= sorted_prices[i + 1]:
                    # 线性插值
                    frac = (last_close - sorted_prices[i]) / (sorted_prices[i + 1] - sorted_prices[i])
                    cdf_at_last = sorted_q[i] + frac * (sorted_q[i + 1] - sorted_q[i])
                    p_up = 1.0 - cdf_at_last
                    break
            if p_up is None:
                p_up = 0.5

    return {
        "ok": True,
        "error": None,
        "model": model_name,
        "horizon": horizon,
        "point_log_return": round(point_log, 5),
        "point_pct": round((math.exp(point_log) - 1) * 100, 2),
        "quantiles_log": quantiles_log,
        "quantiles_pct": quantiles_pct,
        "p_up_native": round(float(p_up), 3) if p_up is not None else None,
        "p_up_5pct_native": round(float(p_up_5), 3) if p_up_5 is not None else None,
        "p_dn_5pct_native": round(float(p_dn_5), 3) if p_dn_5 is not None else None,
        "n_samples": n_samples,
    }


# =============================================================================
# Adapter for conformal.conformalize_external_forecaster
# =============================================================================
def chronos_point_forecaster(close_history: np.ndarray, horizon: int) -> float:
    """
    返回 horizon 期累计 log return 的 *点预测* (中位数).
    供 conformal.conformalize_external_forecaster 使用.
    抛异常 -> 上层捕获后跳过该校准点.
    """
    out = predict_distribution(close_history, horizon=horizon)
    if not out["ok"]:
        raise RuntimeError(out["error"])
    return float(out["point_log_return"])
