# Exceptional Olympic Performance: README

## Overview
This project aims to analyze Olympic medal counts using statistical models to adjust for population size. By assuming every person in the world has an equal chance of winning a medal, we can identify countries with the most statistically significant performances relative to their population. The results highlight exceptional countries with a "WTF, that country is doing well given their population" moment.

## Methodology
### Data Collection
We fetch the latest Olympic medal data from Wikipedia and population data from a local CSV file. The medal table is updated periodically using a caching mechanism to ensure the data is recent without overwhelming the server with frequent requests.

### Statistical Models
1. **Poisson Distribution**
   - The Poisson distribution is used to model the probability of winning a certain number of medals. The probability that a country wins at least `k` medals, given its population, is calculated using the cumulative distribution function (CDF) of the Poisson distribution.
   - The Poisson percent-point function (PPF) is used to find the equivalent number of medals a country would win if it had the same population as the average country the olympic committee.
