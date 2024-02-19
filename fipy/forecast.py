from scipy.stats import rv_histogram
import math

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



