"""
Empirical Historical Frequencies (21 years, 35 stocks, 176K observations)
==========================================================================
Source: Multi-signal event study on US equities 2005-2026.

Each entry contains:
- n: number of historical events found in full 21-year sample
- pct_up_10d: % up 10d after event in full sample
- mean_10d: mean 10-day return after event in full sample
- alpha_10d: difference from baseline
- p_value: statistical significance in full 21-year sample
- significant: bool
- stability: out-of-sample test verdict (TRAIN 2005-2019 vs TEST 2020-2026)
- test_pct_up_10d: % up 10d in TEST period only (recent 6 years)
- oos_significant: was the signal still statistically significant in TEST?

The stability and test_pct_up_10d fields are CRITICAL — they show whether
the 21-year frequency held in recent data.
"""

# Baseline (no signal filter)
BASELINE_10D = {
    "n": 176330,
    "pct_up_5d": 55.1,
    "pct_up_10d": 56.7,
    "pct_up_20d": 58.6,
    "mean_5d": 0.275,
    "mean_10d": 0.543,
    "mean_20d": 1.083,
    # Train/test specific baselines
    "train_pct_up_10d": 57.2,  # 2005-2019
    "test_pct_up_10d": 55.7,   # 2020-2026
}


SIGNAL_STATS = {
    # === Cross signals ===
    "macd_gc": {
        "label": "MACD Golden Cross",
        "category": "cross",
        "n": 7058, "pct_up_10d": 56.6, "mean_10d": 0.53,
        "alpha_10d": -0.01, "p_value": 0.871, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 54.8, "oos_significant": False,
    },
    "macd_dc": {
        "label": "MACD Death Cross",
        "category": "cross",
        "n": 7047, "pct_up_10d": 56.3, "mean_10d": 0.46,
        "alpha_10d": -0.09, "p_value": 0.179, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 54.6, "oos_significant": False,
    },
    "ma_short_gc": {
        "label": "Short MA Cross Up (5/20)",
        "category": "cross",
        "n": 102578, "pct_up_10d": 56.7, "mean_10d": 0.44,
        "alpha_10d": -0.10, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.6, "oos_significant": True,
    },
    "ma_short_dc": {
        "label": "Short MA Cross Down (5/20)",
        "category": "cross",
        "n": 5457, "pct_up_10d": 55.9, "mean_10d": 0.42,
        "alpha_10d": -0.13, "p_value": 0.085, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 54.1, "oos_significant": False,
    },
    "ma_mid_gc": {
        "label": "Mid MA Cross Up (20/50)",
        "category": "cross",
        "n": 107509, "pct_up_10d": 56.1, "mean_10d": 0.42,
        "alpha_10d": -0.12, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 54.8, "oos_significant": True,
    },
    "ma_mid_dc": {
        "label": "Mid MA Cross Down (20/50)",
        "category": "cross",
        "n": 1943, "pct_up_10d": 56.9, "mean_10d": 0.60,
        "alpha_10d": 0.05, "p_value": 0.652, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 54.4, "oos_significant": False,
    },
    "ma_long_gc": {
        "label": "Long MA Cross Up (50/200, classic Golden Cross)",
        "category": "cross",
        "n": 120828, "pct_up_10d": 56.9, "mean_10d": 0.49,
        "alpha_10d": -0.05, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.7, "oos_significant": True,
    },
    "ma_long_dc": {
        "label": "Long MA Cross Down (50/200, classic Death Cross)",
        "category": "cross",
        "n": 507, "pct_up_10d": 61.5, "mean_10d": 1.24,
        "alpha_10d": 0.70, "p_value": 0.006, "significant": True,
        "stability": "stable", "test_pct_up_10d": 61.4, "oos_significant": True,
        "note": "Pattern had positive alpha in historical data — opposite of common assumption; frequencies stable across periods",
    },
    "kdj_gc": {
        "label": "KDJ Cross Up (any)",
        "category": "cross",
        "n": 90046, "pct_up_10d": 56.7, "mean_10d": 0.49,
        "alpha_10d": -0.06, "p_value": 0.001, "significant": True,
        "stability": "stable", "test_pct_up_10d": 56.1, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "kdj_dc": {
        "label": "KDJ Cross Down (any)",
        "category": "cross",
        "n": 22701, "pct_up_10d": 57.0, "mean_10d": 0.57,
        "alpha_10d": 0.03, "p_value": 0.398, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 55.7, "oos_significant": False,
    },
    "kdj_gc_oversold": {
        "label": "KDJ Cross Up in Oversold Zone",
        "category": "cross",
        "n": 9499, "pct_up_10d": 56.4, "mean_10d": 0.65,
        "alpha_10d": 0.11, "p_value": 0.090, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 54.8, "oos_significant": False,
    },
    "kdj_dc_overbought": {
        "label": "KDJ Cross Down in Overbought Zone",
        "category": "cross",
        "n": 9797, "pct_up_10d": 57.4, "mean_10d": 0.44,
        "alpha_10d": -0.10, "p_value": 0.027, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.8, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "dmi_gc": {
        "label": "DMI +DI/-DI Cross Up",
        "category": "cross",
        "n": 97089, "pct_up_10d": 56.6, "mean_10d": 0.46,
        "alpha_10d": -0.08, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.3, "oos_significant": True,
    },
    "dmi_dc": {
        "label": "DMI +DI/-DI Cross Down",
        "category": "cross",
        "n": 8942, "pct_up_10d": 55.3, "mean_10d": 0.36,
        "alpha_10d": -0.19, "p_value": 0.001, "significant": True,
        "stability": "drifted_notable", "test_pct_up_10d": 53.0, "oos_significant": True,
    },
    "dmi_gc_strong": {
        "label": "DMI Cross Up with Strong Trend (ADX > 25)",
        "category": "cross",
        "n": 65143, "pct_up_10d": 57.1, "mean_10d": 0.49,
        "alpha_10d": -0.05, "p_value": 0.005, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 56.0, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "dmi_dc_strong": {
        "label": "DMI Cross Down with Strong Trend (ADX > 25)",
        "category": "cross",
        "n": 4069, "pct_up_10d": 54.5, "mean_10d": 0.32,
        "alpha_10d": -0.22, "p_value": 0.007, "significant": True,
        "stability": "drifted_notable", "test_pct_up_10d": 51.9, "oos_significant": True,
    },
    "psar_flip_up": {
        "label": "Parabolic SAR Flip Up",
        "category": "cross",
        "n": 98096, "pct_up_10d": 57.0, "mean_10d": 0.49,
        "alpha_10d": -0.05, "p_value": 0.001, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 56.0, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "psar_flip_dn": {
        "label": "Parabolic SAR Flip Down",
        "category": "cross",
        "n": 8488, "pct_up_10d": 56.9, "mean_10d": 0.56,
        "alpha_10d": 0.02, "p_value": 0.773, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 55.3, "oos_significant": False,
    },

    # === Support / Resistance proximity ===
    "near_sma200": {
        "label": "Price Near 200-day MA (within 2%)",
        "category": "sr",
        "n": 22310, "pct_up_10d": 56.7, "mean_10d": 0.46,
        "alpha_10d": -0.08, "p_value": 0.008, "significant": True,
        "stability": "drifted_notable", "test_pct_up_10d": 53.7, "oos_significant": True,
    },
    "near_swing_high": {
        "label": "Price Near 60-day High (within 2%)",
        "category": "sr",
        "n": 49200, "pct_up_10d": 57.5, "mean_10d": 0.39,
        "alpha_10d": -0.16, "p_value": 0.0000, "significant": True,
        "stability": "stable", "test_pct_up_10d": 56.9, "oos_significant": True,
        "note": "Frequencies were similar across TRAIN and TEST periods",
    },
    "near_swing_low": {
        "label": "Price Near 60-day Low (within 2%)",
        "category": "sr",
        "n": 13906, "pct_up_10d": 57.8, "mean_10d": 0.83,
        "alpha_10d": 0.29, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.9, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "near_bb_upper": {
        "label": "Price Near Bollinger Upper Band",
        "category": "sr",
        "n": 34126, "pct_up_10d": 57.1, "mean_10d": 0.36,
        "alpha_10d": -0.19, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.9, "oos_significant": True,
    },
    "near_bb_lower": {
        "label": "Price Near Bollinger Lower Band",
        "category": "sr",
        "n": 21556, "pct_up_10d": 56.6, "mean_10d": 0.68,
        "alpha_10d": 0.14, "p_value": 0.0003, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.2, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "near_r1": {
        "label": "Price Near Pivot R1",
        "category": "sr",
        "n": 49788, "pct_up_10d": 57.3, "mean_10d": 0.47,
        "alpha_10d": -0.08, "p_value": 0.0001, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.5, "oos_significant": True,
    },
    "near_s1": {
        "label": "Price Near Pivot S1",
        "category": "sr",
        "n": 43883, "pct_up_10d": 57.0, "mean_10d": 0.50,
        "alpha_10d": -0.04, "p_value": 0.043, "significant": True,
        "stability": "drifted_notable", "test_pct_up_10d": 54.8, "oos_significant": True,
    },

    # === Candle patterns ===
    "pat_doji": {
        "label": "Doji",
        "category": "pattern",
        "n": 18724, "pct_up_10d": 56.9, "mean_10d": 0.57,
        "alpha_10d": 0.03, "p_value": 0.409, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 55.5, "oos_significant": False,
    },
    "pat_hammer": {
        "label": "Hammer (small body, long lower shadow)",
        "category": "pattern",
        "n": 8010, "pct_up_10d": 55.3, "mean_10d": 0.33,
        "alpha_10d": -0.22, "p_value": 0.0003, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 54.4, "oos_significant": True,
        "note": "Pattern had negative alpha in historical data — opposite of common assumption",
    },
    "pat_inverted_hammer": {
        "label": "Inverted Hammer (small body, long upper shadow)",
        "category": "pattern",
        "n": 6817, "pct_up_10d": 58.7, "mean_10d": 0.86,
        "alpha_10d": 0.32, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 57.5, "oos_significant": True,
        "note": "Pattern was statistically significant in both TRAIN and TEST periods",
    },
    "pat_bull_engulf": {
        "label": "Bullish Engulfing",
        "category": "pattern",
        "n": 6111, "pct_up_10d": 56.3, "mean_10d": 0.42,
        "alpha_10d": -0.13, "p_value": 0.043, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.3, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "pat_bear_engulf": {
        "label": "Bearish Engulfing",
        "category": "pattern",
        "n": 6907, "pct_up_10d": 58.2, "mean_10d": 0.81,
        "alpha_10d": 0.26, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 57.2, "oos_significant": True,
        "note": "Pattern had positive alpha in historical data — opposite of common assumption",
    },
    "pat_three_white": {
        "label": "Three White Soldiers (3 consecutive up days)",
        "category": "pattern",
        "n": 16315, "pct_up_10d": 56.5, "mean_10d": 0.37,
        "alpha_10d": -0.18, "p_value": 0.0000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.2, "oos_significant": True,
    },
    "pat_three_black": {
        "label": "Three Black Crows (3 consecutive down days)",
        "category": "pattern",
        "n": 12628, "pct_up_10d": 56.6, "mean_10d": 0.69,
        "alpha_10d": 0.14, "p_value": 0.008, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 54.7, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },

    # === Advanced chart patterns (newly added, 21Y validated) ===
    "pat_double_bottom": {
        "label": "Double Bottom (W shape)",
        "category": "advanced_pattern",
        "n": 5832, "pct_up_10d": 68.3, "mean_10d": 2.00,
        "alpha_10d": 1.45, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 67.3, "oos_significant": True,
        "note": "21-year alpha was +1.45%; pattern was statistically significant in both TRAIN and TEST periods",
    },
    "pat_double_top": {
        "label": "Double Top (M shape)",
        "category": "advanced_pattern",
        "n": 6088, "pct_up_10d": 43.1, "mean_10d": -1.08,
        "alpha_10d": -1.63, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 42.3, "oos_significant": True,
        "note": "21-year alpha was -1.63%; pattern was statistically significant in both TRAIN and TEST periods",
    },
    "pat_head_shoulders": {
        "label": "Head and Shoulders",
        "category": "advanced_pattern",
        "n": 1515, "pct_up_10d": 41.8, "mean_10d": -1.27,
        "alpha_10d": -1.82, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 39.0, "oos_significant": True,
        "note": "21-year alpha was negative; smaller sample size (n=1.5K)",
    },
    "pat_inverse_head_shoulders": {
        "label": "Inverse Head and Shoulders",
        "category": "advanced_pattern",
        "n": 1943, "pct_up_10d": 68.7, "mean_10d": 2.09,
        "alpha_10d": 1.54, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 67.5, "oos_significant": True,
        "note": "21-year alpha was positive in both TRAIN and TEST periods",
    },
    "pat_uptrend_channel": {
        "label": "Uptrend Channel (40-day, R²>0.5)",
        "category": "advanced_pattern",
        "n": 57595, "pct_up_10d": 55.9, "mean_10d": 0.42,
        "alpha_10d": -0.13, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 53.9, "oos_significant": True,
        "note": "Pattern had slight negative alpha in historical data — opposite of common assumption",
    },
    "pat_downtrend_channel": {
        "label": "Downtrend Channel (40-day, R²>0.5)",
        "category": "advanced_pattern",
        "n": 26573, "pct_up_10d": 58.2, "mean_10d": 0.88,
        "alpha_10d": 0.33, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 60.3, "oos_significant": True,
        "note": "Pattern had positive alpha in historical data — opposite of common assumption",
    },
    "pat_triangle_symmetric": {
        "label": "Symmetric Triangle Consolidation",
        "category": "advanced_pattern",
        "n": 252, "pct_up_10d": 45.6, "mean_10d": -1.80,
        "alpha_10d": -2.35, "p_value": 0.001, "significant": True,
        "stability": "drifted_notable", "test_pct_up_10d": 41.2, "oos_significant": True,
        "note": "Small sample size (n=252)",
    },
    "pat_triangle_ascending": {
        "label": "Ascending Triangle",
        "category": "advanced_pattern",
        "n": 2718, "pct_up_10d": 58.0, "mean_10d": 0.76,
        "alpha_10d": 0.21, "p_value": 0.023, "significant": True,
        "stability": "drifted_notable", "test_pct_up_10d": 52.4, "oos_significant": False,
        "note": "21-year sample showed p<0.05 but TEST period showed p>=0.05",
    },
    "pat_triangle_descending": {
        "label": "Descending Triangle",
        "category": "advanced_pattern",
        "n": 2170, "pct_up_10d": 55.8, "mean_10d": 0.53,
        "alpha_10d": -0.02, "p_value": 0.859, "significant": False,
        "stability": "drifted_notable", "test_pct_up_10d": 49.2, "oos_significant": False,
    },
    "pat_breakout_up": {
        "label": "Breakout Up (close > 20-day high)",
        "category": "advanced_pattern",
        "n": 29294, "pct_up_10d": 57.3, "mean_10d": 0.42,
        "alpha_10d": -0.13, "p_value": 0.000, "significant": True,
        "stability": "stable", "test_pct_up_10d": 56.6, "oos_significant": True,
    },
    "pat_breakout_down": {
        "label": "Breakout Down (close < 20-day low)",
        "category": "advanced_pattern",
        "n": 16901, "pct_up_10d": 57.7, "mean_10d": 0.99,
        "alpha_10d": 0.44, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 56.1, "oos_significant": True,
        "note": "Pattern had positive alpha in historical data — opposite of common assumption",
    },
    "pat_morning_star": {
        "label": "Morning Star (3-bar bullish reversal)",
        "category": "advanced_pattern",
        "n": 2266, "pct_up_10d": 55.7, "mean_10d": 0.34,
        "alpha_10d": -0.21, "p_value": 0.063, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 54.6, "oos_significant": False,
    },
    "pat_evening_star": {
        "label": "Evening Star (3-bar bearish reversal)",
        "category": "advanced_pattern",
        "n": 2210, "pct_up_10d": 57.6, "mean_10d": 0.68,
        "alpha_10d": 0.13, "p_value": 0.216, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 56.5, "oos_significant": False,
    },

    # === Extra chart patterns (21Y validated, real backtest) ===
    "pat_rising_wedge": {
        "label": "Rising Wedge",
        "category": "advanced_pattern",
        "n": 9849, "pct_up_10d": 58.0, "mean_10d": 0.44,
        "alpha_10d": -0.10, "p_value": 0.015, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.3, "oos_significant": True,
    },
    "pat_falling_wedge": {
        "label": "Falling Wedge",
        "category": "advanced_pattern",
        "n": 1472, "pct_up_10d": 55.8, "mean_10d": 0.88,
        "alpha_10d": 0.33, "p_value": 0.021, "significant": True,
        "stability": "stable", "test_pct_up_10d": 55.7, "oos_significant": False,
    },
    "pat_bull_flag": {
        "label": "Bull Flag",
        "category": "advanced_pattern",
        "n": 1193, "pct_up_10d": 54.7, "mean_10d": 0.47,
        "alpha_10d": -0.08, "p_value": 0.570, "significant": False,
        "stability": "stable", "test_pct_up_10d": 55.6, "oos_significant": False,
    },
    "pat_bear_flag": {
        "label": "Bear Flag",
        "category": "advanced_pattern",
        "n": 547, "pct_up_10d": 53.4, "mean_10d": 0.38,
        "alpha_10d": -0.17, "p_value": 0.397, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 55.3, "oos_significant": False,
    },
    "pat_pennant": {
        "label": "Pennant",
        "category": "advanced_pattern",
        "n": 237, "pct_up_10d": 56.5, "mean_10d": 1.93,
        "alpha_10d": 1.38, "p_value": 0.050, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 58.6, "oos_significant": True,
        "note": "Small sample (n=237); statistically borderline (p=0.050)",
    },
    "pat_cup_handle": {
        "label": "Cup and Handle",
        "category": "advanced_pattern",
        "n": 6357, "pct_up_10d": 53.8, "mean_10d": 0.15,
        "alpha_10d": -0.39, "p_value": 0.000, "significant": True,
        "stability": "drifted_minor", "test_pct_up_10d": 55.4, "oos_significant": True,
        "note": "Pattern had negative alpha in historical data — opposite of common assumption",
    },
    "pat_rounding_bottom": {
        "label": "Rounding Bottom",
        "category": "advanced_pattern",
        "n": 4308, "pct_up_10d": 56.6, "mean_10d": 0.49,
        "alpha_10d": -0.05, "p_value": 0.480, "significant": False,
        "stability": "stable", "test_pct_up_10d": 56.1, "oos_significant": True,
    },
    "pat_gap_up": {
        "label": "Gap Up (>2%)",
        "category": "advanced_pattern",
        "n": 1438, "pct_up_10d": 53.3, "mean_10d": 0.58,
        "alpha_10d": 0.04, "p_value": 0.863, "significant": False,
        "stability": "drifted_notable", "test_pct_up_10d": 56.6, "oos_significant": True,
    },
    "pat_gap_down": {
        "label": "Gap Down (>2%)",
        "category": "advanced_pattern",
        "n": 1597, "pct_up_10d": 55.2, "mean_10d": 0.66,
        "alpha_10d": 0.12, "p_value": 0.661, "significant": False,
        "stability": "stable", "test_pct_up_10d": 55.2, "oos_significant": False,
    },
    "pat_inside_bar": {
        "label": "Inside Bar (today's range inside yesterday's)",
        "category": "advanced_pattern",
        "n": 22248, "pct_up_10d": 56.5, "mean_10d": 0.53,
        "alpha_10d": -0.01, "p_value": 0.782, "significant": False,
        "stability": "stable", "test_pct_up_10d": 55.8, "oos_significant": False,
    },
    "pat_outside_bar": {
        "label": "Outside Bar (today's range engulfs yesterday's)",
        "category": "advanced_pattern",
        "n": 17372, "pct_up_10d": 57.1, "mean_10d": 0.60,
        "alpha_10d": 0.05, "p_value": 0.173, "significant": False,
        "stability": "drifted_minor", "test_pct_up_10d": 55.8, "oos_significant": False,
    },
}


def get_signal_stat(signal_key):
    return SIGNAL_STATS.get(signal_key)


def stability_badge(stat: dict) -> str:
    """
    Return a neutral description of how the signal's historical frequency
    behaved across time periods. NOT a quality rating. NOT a recommendation.
    """
    if not stat:
        return ""
    s = stat.get("stability", "unknown")
    oos = stat.get("oos_significant", False)
    test_pct = stat.get("test_pct_up_10d")
    pct = stat.get("pct_up_10d")

    if test_pct is None or pct is None:
        return "data limited"

    drift = test_pct - pct
    if abs(drift) < 1.0:
        drift_desc = f"recent 6yr matched 21yr ({drift:+.1f}%)"
    elif abs(drift) < 3.0:
        drift_desc = f"minor drift in recent 6yr ({drift:+.1f}%)"
    elif abs(drift) < 5.0:
        drift_desc = f"notable drift in recent 6yr ({drift:+.1f}%)"
    else:
        drift_desc = f"large drift in recent 6yr ({drift:+.1f}%)"

    oos_desc = "p<0.05 in recent 6yr" if oos else "p>=0.05 in recent 6yr"
    return f"{drift_desc}; {oos_desc}"


def format_freq_text(stat: dict) -> str:
    if not stat:
        return ""
    n = stat["n"]
    pct = stat["pct_up_10d"]
    test_pct = stat.get("test_pct_up_10d", pct)
    mean = stat["mean_10d"]
    alpha = stat["alpha_10d"]
    p = stat["p_value"]

    text = f"21-year frequency: in {n:,} similar events, {pct:.1f}% saw a higher price 10 days later "
    text += f"(baseline: 56.7%). "
    text += f"In recent 6 years (2020-2026): {test_pct:.1f}%."
    if "note" in stat:
        text += f"\n  Note: {stat['note']}"
    return text
