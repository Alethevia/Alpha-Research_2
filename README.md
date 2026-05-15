# Stock Indicator Historical Frequency Viewer

> **EDUCATIONAL & RESEARCH PROJECT — NOT INVESTMENT ADVICE**

An open-source research tool that displays:
- Currently triggered technical signals for a given stock symbol
- Historical frequency statistics from a 21-year backtest study
- Public earnings information

## Disclaimer

This tool is provided for educational and research purposes only.

- This tool does **not** predict future prices
- This tool does **not** provide investment advice
- This tool does **not** recommend buying or selling any security
- This tool does **not** rate stocks as good, bad, or any quality level

The author is not a registered investment advisor (RIA). All trading decisions
are the sole responsibility of the user. Past frequencies do not predict future
behavior. By using this tool, you accept all risks.

## What it shows

For an input stock symbol:

1. **Currently detected technical events** (objective measurements)
2. **Historical frequencies** of how the same events behaved in 21 years of US equity data
3. **Out-of-sample validation** (TRAIN 2005-2019 vs TEST 2020-2026)
4. **Public earnings data** (no rating attached)
5. **Charts** with annotation of detected events

## Features

### Signal Detection

- 11 cross-type signals (MACD, MA short/mid/long, KDJ, DMI, PSAR)
- 7 support/resistance proximity flags
- 7 basic candlestick patterns
- 14 advanced chart patterns (double bottom/top, head and shoulders, channels, etc.)
- 8 additional chart patterns (wedges, flags, pennants, etc.)

### Validation Statistics (Embedded)

Every signal carries historical statistics:
- Sample size in 21-year backtest
- 21-year frequency (% up after N days)
- Recent 6-year frequency (% up in TEST period)
- Statistical significance (p-value) in full sample
- Statistical significance in out-of-sample TEST period

### User Interface

- **Single Stock Analysis** tab: detailed view of one symbol
- **Multi-Timeframe Confluence** tab: Daily/Weekly/Monthly MACD and MA states
- **Pool Scanner** tab: scan multiple symbols at once

## How to Run

### Locally

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Deploy (Free Cloud)

#### Option 1: Hugging Face Spaces
1. Register account at huggingface.co
2. Create new Space → SDK: Streamlit
3. Upload all `.py` files and `requirements.txt`
4. Auto-deploys in 1-2 minutes

#### Option 2: Streamlit Community Cloud
1. Push code to GitHub
2. Connect at share.streamlit.io
3. Auto-deploys

## Methodology Notes

### Backtest

- Period: 2005-04-01 to 2026-04-08 (~21 years)
- Universe: 35 US large-cap equities listed before 2005 (controls for survivorship bias)
- Observations: ~185,000
- Look-ahead bias: controlled via `merge_asof` with backward direction
- Friction: 5 bps per side modeled in `_net` columns

### Statistical Validation

- t-test against unconditional baseline
- TRAIN/TEST split for out-of-sample testing
- Bootstrap 95% confidence intervals (advanced pattern validation)
- Bonferroni correction for multiple testing (advanced pattern validation)
- Direction consistency check across timeframes

### Reported Findings (Honest Disclosure)

- Average frequency drift between TRAIN and TEST periods: ~2.15 percentage points
- 8 of 32 indicator signals lost statistical significance in TEST period
- Strongest historical alpha appeared in classical chart patterns (double bottom/top,
  head and shoulders) at monthly timeframe — but these events are rare (dozens per year
  across 35 stocks)
- Most indicator-based confluence combinations did not improve predictive power over
  single signals

These findings are openly disclosed because educational tools should not overstate
their accuracy. Effect sizes are typically small (< 1% per 10 days) and may not
exceed transaction costs in practice.

## Files

- `app.py` — Main Streamlit application
- `extended_signals.py` — Cross signals, S/R proximity, basic candle patterns
- `advanced_patterns.py` — Double bottom/top, head and shoulders, channels, triangles, breakouts, star patterns
- `extra_patterns.py` — Wedges, flags, pennants, cup and handle, rounding bottom, gaps, inside/outside bars
- `historical_stats.py` — Embedded 21-year backtest statistics
- `requirements.txt` — Python dependencies

## Indicator References

- MACD: Appel, G. (1979). *Systems and Forecasts*
- KDJ / Stochastic: Lane, G. (1950s)
- DMI / ADX: Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*
- PSAR: Wilder, J. W. (1978)
- Bollinger Bands: Bollinger, J. (2001)
- Candlestick patterns: Nison, S. (1991). *Japanese Candlestick Charting Techniques*
- Chart patterns: Edwards, R. D., & Magee, J. (1948). *Technical Analysis of Stock Trends*
- Pivot Points: standard floor trader pivots

## License

MIT License — free to use, modify, and distribute, with the condition that all
disclaimers must be preserved in derivative works.

## Project

Open-source research project. No author identification. No commercial entity
associated. No fees collected. No personalized advice provided.
