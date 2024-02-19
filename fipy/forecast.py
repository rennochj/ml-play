from scipy.stats import rv_histogram
import math

# -- outflow ratio ----------------------------------------------------------------------------------------------------

def outflow_ratio(dist: rv_histogram, epsilon: float, inflation: float, periods: int, iterations: int) -> list[float]:

    gamma: list[float] = []

    for _ in range(1, iterations):

        rates = dist.rvs(size=periods)

        kappa = 1.0
        rho = rates[0]

        for i in range(periods):
            rho *= rates[i]
            kappa = kappa * rates[i] + inflation ** i

        gamma.append( (rho - epsilon)/kappa )

    return gamma

# -- outflow ratio ----------------------------------------------------------------------------------------------------

def outflow_ratio(dist: rv_histogram, initial_value: float, outflow: float, inflation: float, periods: int, iterations: int) -> list[float]:

    for _ in range(1, iterations):

        rates: rv_histogram = dist.rvs(size=periods)

        final_values: list[float] = [initial_value]

        for i in range(periods):
            value = rates[i] * value - outflow * inflation ** i

            if value < 0.0:
                value = 0.0
                exit

        final_values.append( value )

    return final_values



