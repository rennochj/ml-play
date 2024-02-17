import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

START_DATE = "1920-1-1"
PERIOD = "1day"

tickers = [
    {
        "name": "spx",
        "symbol": "^SPX"
    },
    {
        "name": "vbmfx",
        "symbol": "VBMFX"
    }
]

fig = make_subplots(rows=len(tickers), cols=1)
row = 1

for ticker in tickers:

    data = yf.download(
        ticker["symbol"], 
        start=START_DATE, 
        period=PERIOD
    )

    data.to_feather(f"data/{ticker['name']}.feather")

    print(f"columns: {ticker['name']}: {data.columns}")
    print(f"index: {ticker['name']}: {data.index}")

    fig.add_trace(
        go.Scatter(y=data["Adj Close"], name=ticker["name"]),
        row=row,
        col=1,
    )

    row += 1

fig.show()


