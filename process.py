import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

N_FORECASTS = 1000
N_FORECAST_PERIODS = 30 * 12 * 21
MAX_RESULT = 30000
INITIAL_VALUE = 2500000
INITIAL_SPLIT = [1.0]
INITIAL_OUTFLOW = 100000.0 / 12
OUTFLOW_RATE = 0. / 12
MOD = 21
N_EXTRACTS = 5

portfolio = [
    {
        "name": "spx",
        "model": "data/spx.feather",
        "current_value": INITIAL_VALUE * INITIAL_SPLIT[0],
        "outflow_split": 1.0
    },
    # {
    #     "name": "vbmfx",
    #     "model": "data/vbmfx.feather",
    #     "current_value": INITIAL_VALUE * INITIAL_SPLIT[1],
    #     "outflow_split": 0.3
    # }
]

def load_models(portfolio):
    models = None
    for investment in portfolio:

        if models is None:
            models = pd.read_feather(investment["model"]).reset_index()[["Date", "Adj Close"]]
        
        else:
            models = pd.merge(
                left=models,
                right=pd.read_feather(investment["model"]).reset_index()[["Date", "Adj Close"]],
                on="Date",
                how="outer"
            )
            
        models.rename(columns={"Adj Close": (investment["name"], "adj close")}, inplace=True)
        models[(investment["name"], "rate")] = models[[(investment["name"], "adj close")]] / models[[(investment["name"], "adj close")]].shift(1)
            
    models.rename(columns={"Date": "date"}, inplace=True)

    return models

def make_distributions(portfolio, models):

    distributions = {}

    for investment in portfolio:
        rates = models[(investment["name"], "rate")].dropna()
        hist = np.histogram(rates, 100, density=True)
        distributions[investment["name"]] = stats.rv_histogram(
            hist,
            density=True
        )

    return distributions

def create_forecast(portfolio, distributions, outflows):

    forecast = pd.DataFrame({"period": np.arange(N_FORECAST_PERIODS+1)})

    forecast["outflow"] = outflows

    value_forecast = { investment["name"]: [investment["current_value"]] for investment in portfolio }
    rates = { investment["name"]: distributions[investment["name"]].rvs(size=N_FORECAST_PERIODS) for investment in portfolio }

    for investment in portfolio:
        forecast[investment["name"] + " rates"] = np.append(rates[investment["name"]], 0)

    short_fall = False

    for i in range(N_FORECAST_PERIODS):

        for investment in portfolio:

            if short_fall:
                value_forecast[investment["name"]].append( np.NaN )

            elif not short_fall and rates[investment["name"]][i] * value_forecast[investment["name"]][i] <= outflows[i] * investment["outflow_split"]:
                value_forecast[investment["name"]].append( np.NaN )
                short_fall = True

            else:
                value_forecast[investment["name"]].append( rates[investment["name"]][i] * value_forecast[investment["name"]][i] - outflows[i] * investment["outflow_split"] )

    for investment in portfolio:
        forecast[investment["name"]] = value_forecast[investment["name"]]

    forecast["total"] = forecast[[investment["name"] for investment in portfolio]].sum(axis=1)

    return forecast

def make_linear_outflows(initial, rate, mod=1, size=N_FORECAST_PERIODS):

    outflow = initial
    outflows = []

    for i in range(N_FORECAST_PERIODS+1):

        if i % mod == 0:
            outflows.append(outflow)
            outflow = outflow * (1.0 + rate)

        else:
            outflows.append(0.0)

    return outflows

if __name__ == '__main__':

    fig = make_subplots(rows=1, cols=2)

    models = load_models(portfolio)

    distributions = make_distributions(portfolio, models)
    results = []

    outflows = make_linear_outflows(INITIAL_OUTFLOW, OUTFLOW_RATE, mod=21)

    for k in range(N_FORECASTS):

        forecast = create_forecast(portfolio, distributions, outflows)

        if k < N_EXTRACTS:
            forecast.to_excel(f"data/forecast-{k}.xlsx")

        results.append(forecast["total"].iloc[-1])

        fig.add_trace(
            go.Scatter(
                y=forecast["total"],
                x=forecast["period"]
            ), 
            row=1, 
            col=1
        )

        if k == 1:
            forecast.to_csv("test.csv")

    result_hist = np.histogram(
        results,
        bins=np.logspace(
            math.floor(math.log10(min(results)+0.1)), 
            math.ceil(math.log10(max(results)+0.1)), 
            num=50, 
            endpoint=True
        ),
        density=False
    )

    # cumm_hist = []
    # cumm = 0.0
    # for value in result_hist[0]:
    #     cumm += value
    #     cumm_hist.append(cumm) 

    # fig.add_trace(
    #     go.Scatter(
    #         x=cumm_hist / result_hist[0].sum(),
    #         y=result_hist[1]
    #     ),
    #     row=1,
    #     col=2
    # )

    fig.add_trace(
        go.Scatter(
            x=result_hist[0],
            y=result_hist[1]
        ),
        row=1,
        col=2
    )

    fig.update_yaxes(
        range=[
            math.floor(math.log10(min(results)+0.1)), 
            math.ceil(math.log10(max(results)+0.1))
        ],
        type="log"
    )

    fig.show()


    print(f"successes: {sum([1 for v in results if v > 1.0])/N_FORECASTS}")

