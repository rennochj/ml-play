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
INITIAL_SPLIT = [0.5, 0.5]
INITIAL_OUTFLOW = 100000.0 / 12 / 21
OUTFLOW_RATE = 0.04 / 12 / 21
MOD = 1
N_EXTRACTS = 5

portfolio = [
    {
        "name": "spx",
        "model": "data/spx.feather",
        "current_value": INITIAL_VALUE * INITIAL_SPLIT[0],
        "outflow_split": 0.5
    },
    {
        "name": "vbmfx",
        "model": "data/vbmfx.feather",
        "current_value": INITIAL_VALUE * INITIAL_SPLIT[1],
        "outflow_split": 0.5
    }
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

def process_growth(current_values, current_rates):
    return current_values * current_rates

def rebalance(current_values, balance):
    total = sum(current_values)
    return balance * total

def process_outflows(current_values, outflows):

    next_values = current_values - outflows
    short_fall = [ 1 if value < 0.0 else 0 for value in next_values ]

    if sum(short_fall) > 0:
        next_values = [np.NAN] * len(next_values)

    return next_values

def create_forecast(portfolio, distributions, outflows):

    forecast = pd.DataFrame({"period": np.arange(N_FORECAST_PERIODS+1)})

    forecast["outflow"] = outflows

    investment_names = []
    rate_names = []
    rates = []

    for investment in portfolio:
        investment_names.append(investment["name"])
        rate_names.append(investment["name"] + "-rates")
        rates.append( distributions[investment["name"]].rvs(size=N_FORECAST_PERIODS+1) )

    rates = np.array(rates).transpose()
    forecast[rate_names] = rates

    outflow_splits = np.array([ investment["outflow_split"] for investment in portfolio ])
    outflow_splits = np.array([ investment["outflow_split"] for investment in portfolio ])

    current_values = np.array([investment["current_value"] for investment in portfolio ])
    values = [current_values]
    
    for i in range(N_FORECAST_PERIODS):
        current_rates = rates[i]
        current_values = process_growth(current_values, current_rates)
        current_values = rebalance(current_values, np.array(INITIAL_SPLIT))
        current_values = process_outflows(current_values, outflow_splits * outflows[i])
        values.append(current_values)

    forecast[investment_names] = values

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

    outflows = make_linear_outflows(INITIAL_OUTFLOW, OUTFLOW_RATE, mod=MOD)

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

