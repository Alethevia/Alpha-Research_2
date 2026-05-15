# Claude Code 完整交接文档
# 股票技术分析工具 - 从 Phase 1 到完整版

## 一句话概述
给同事的金融技术分析 web 工具，部署在 HuggingFace Spaces (Docker + Streamlit)。
目标: $5/月成本覆盖 TradingView Premium 90% + AI 能力(TV没有)。
URL: https://huggingface.co/spaces/AlphaResearch2026/stock-indicator-viewer

## 当前文件 (10个, 3134行Python)
- app.py (847行): 主程序, 中文UI, 已整合所有模块
- extended_signals.py (255行): 32个基础信号
- advanced_patterns.py (309行): 14个高级形态
- extra_patterns.py (273行): 11个扩展形态  
- historical_stats.py (513行): 56个信号21年回测统计
- new_indicators.py (547行): RSI/CCI/WR/OBV/ATR/VWAP/Supertrend/Ichimoku/NXCD/TRIX
- confluence.py (390行): 6维共振评分
- requirements.txt: streamlit,pandas,numpy,yfinance,matplotlib,scipy
- README.md: HuggingFace YAML配置+说明

## 已完成功能
1. 56个旧信号 + 40个新指标信号 = 96个信号
2. NXCD(NX多空通道 + CD MACD背离)
3. 6维共振评分(趋势/动量/超买超卖/成交量/波动率/形态)
4. 背离检测(RSI/MACD/OBV三种)
5. 中文UI + 英文术语 + 96个信号中文翻译
6. Yahoo符号搜索
7. 多周期分析(日/周/月)
8. 股票池扫描
9. 双语免责声明

## Claude Code 需要做的任务 (按优先级)

### 任务1: Plotly交互式K线图 (最高优先级)
替代matplotlib静态图。用plotly.graph_objects + make_subplots。
- 主图: Candlestick + 可叠加SMA/EMA/Bollinger/Ichimoku云/Supertrend
- 副图: MACD/RSI/成交量
- 用户checkbox选择叠加哪些指标
- 信号标记(金叉/死叉/双底/背离)在K线上标注
- requirements.txt 加 plotly

### 任务2: Kronos-mini概率预测
模型: NeoQuasar/Kronos-mini (4.1M参数, MIT, CPU可跑)
或替代: amazon/chronos-bolt-tiny (9M, Apache 2.0)
输入OHLCV → 输出未来K线概率分布
显示: "N日后价格>当前概率: XX% (95%CI: YY%-ZZ%)"
必须加免责声明

### 任务3: Conformal Prediction校准
给概率输出加数学保证的置信区间
用split conformal或MAPIE库

### 任务4: 自动最优窗口
测试30/60/90/180/365天窗口
选共振评分最稳定的窗口

### 任务5: Finnhub免费版整合
pip install finnhub-python
新闻+情绪, 内幕交易, 分析师评级
免费版60次/分, 不需要付费

### 任务6: AI解读层(需要API key)
GPT-5.4 Nano(自动摘要) + Mini(追问) + Claude Haiku(新闻搜索)
只在用户点按钮时调用, 不是每次刷新

### 任务7: TTM Squeeze/Chaikin Money Flow/Keltner/Heikin-Ashi
四个额外指标加到new_indicators.py

### 任务8: Gradio迁移(可选)
如果Streamlit继续抖动, 迁移到Gradio

## 代码质量强制要求
- 无占位符(TODO/FIXME/pass/mock/dummy)
- 中文UI, 英文术语, matplotlib/plotly label必须英文(HF无中文字体)
- Streamlit widget key必须唯一
- 不能用 if btn or input: (抖动), 用 if btn:
- 时区: datetime.now(timezone.utc)
- pd.Timestamp()不带tz="UTC"(如果已tz-aware)
- .shift(1).fillna(False)后加.astype(bool)
- 除法前检查除零
- plt.close(fig)避免内存泄漏
- except Exception后必须有合理处理

## 法律强制
- 红色双语免责在最顶部
- 禁止: 建议买入/卖出, 推荐, should buy/sell, recommend
- 共振评分旁必须免责
- 概率输出标注"不构成投资建议"
- 基于Lowe v. SEC(1985)出版商豁免

## 部署
HuggingFace Space: AlphaResearch2026/stock-indicator-viewer
SDK: Docker(Streamlit template), app_port: 8501
Dockerfile已配置, ENTRYPOINT指向app.py
上传: 用huggingface_hub Python库 upload_file

## 回测背景
21年(2005-2026), 35只美股, ~185K观测
TRAIN/TEST分割, 平均漂移2.15%
最强alpha: 图表形态±1.5%(但极小, 可能不够覆盖交易成本)
多指标共振未显著提升预测力
