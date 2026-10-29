import numpy as np
import pandas as pd
from historical_stats import SIGNAL_STATS, BASELINE_10D
from extended_signals import compute_all_signals
from advanced_patterns import detect_all_advanced_patterns
from extra_patterns import detect_extra_patterns
from conformal import probability_forecast

def test_unverified_statistics_are_not_published():
    assert SIGNAL_STATS
    assert not BASELINE_10D
    for value in SIGNAL_STATS.values():
        assert value['verified'] is False
        assert not ({'p_value','n','alpha_10d','pct_up_10d'} & value.keys())

def test_signal_and_forecast_pipeline_on_synthetic_fixture():
    rng = np.random.default_rng(7)
    close = 100*np.exp(np.cumsum(rng.normal(0,0.01,600)))
    df = pd.DataFrame(dict(open=close, high=close*1.01, low=close*0.99, close=close, volume=10000.), index=pd.date_range('2020-01-01',periods=600))
    out = detect_extra_patterns(detect_all_advanced_patterns(compute_all_signals(df)))
    assert len(out)==600
    assert 'macd_gc' in out
    result = probability_forecast(out, horizon=5)
    assert isinstance(result, dict)
    assert result['ok'] is True
