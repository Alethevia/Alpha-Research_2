# Stock signal explorer

A Streamlit interface for calculating technical signals and charting supplied/downloaded stock prices. A Gradio interface and model-based forecasting experiments are optional.

**Historical claims withdrawn:** the old embedded 21-year frequencies, Alpha values, p-values and out-of-sample significance labels did not include their underlying dataset and reproducible event-study runner. They have been removed from the catalogue and both interfaces. Signal names remain; absence of verified statistics is explicit. Selecting surviving companies that existed before 2005 does not remove survivorship bias.

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Optional Gradio / Chronos dependencies are separate:

```bash
python -m pip install -r requirements-optional.txt
python gradio_app.py
```

Market downloads require network access. Finnhub and LLM sections require their own configured credentials. Forecast ranges and heuristic confidence scores are experimental, not guaranteed coverage or calibrated probabilities. No broker execution is performed by the UI. See [AUDIT.md](AUDIT.md) for what was actually checked.
