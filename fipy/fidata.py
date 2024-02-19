import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf

# -- fetch_tickers ----------------------------------------------------------------------------------------------------

def fetch_tickers(tickers: list[dict], path: str="data", rate_column: str="Adj Close") -> None:

    for ticker in tickers:
        data = yf.download(
            ticker["symbol"], 
            start=ticker["start"], 
            period=ticker["period"]
        )

        data.to_feather(f"{path}/{ticker['name']}.feather")

# -- load models ----------------------------------------------------------------------------------------------------

def load_model(path: str, rate_column: str="Adj Close", bins: int=100) -> stats.rv_histogram:

    data = pd.read_feather("data/vbmfx.feather").reset_index()
    data["rate"] = data["Adj Close"] / data["Adj Close"].shift(1)
    data["rate"] = data["rate"].fillna(1.0)

    hist = np.histogram(data["rate"], bins, density=True)
    return stats.rv_histogram(hist, density=True)