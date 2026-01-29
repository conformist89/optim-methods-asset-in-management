# Portfolio Optimization and Risk Analysis

This repository contains a quantitative finance project focused on portfolio optimization and risk analysis within the mean–variance framework. The project demonstrates practical implementation of classical portfolio theory, Sharpe ratio optimization, and tail risk measures using real equity return data.

In addition, the repository includes solutions to selected assignments from the Coursera course *Optimization Methods in Asset Management*, which are organized separately.

---

## Project overview

The main analysis investigates portfolio construction and risk characteristics for a universe of 23 U.S. stocks, using daily returns over the period:

- January 4, 2016 – December 28, 2017  
- Data source: Yahoo Finance

The project includes the following components:

- Estimation of expected returns and covariance matrix
- Visualization of correlation structure and return distributions
- Mean–variance portfolio optimization under return constraints
- Construction of the efficient frontier
- Sharpe ratio (tangency portfolio) optimization
- Risk analysis using Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- Comparison between:
  - an equally weighted portfolio
  - a Sharpe-optimal portfolio

The analysis highlights the relationship between risk-adjusted performance and tail risk, illustrating cases where higher Sharpe ratios coincide with worse VaR and CVaR outcomes.

---

## Key findings

- The Sharpe-optimal portfolio achieves a higher mean return than the equally weighted benchmark.
- Despite improved risk-adjusted performance, the Sharpe-optimal portfolio exhibits larger VaR and CVaR at the same confidence level.
- This occurs because Sharpe ratio optimization does not directly constrain tail losses or downside risk.
- Equal-weighted portfolios, while suboptimal in a mean–variance sense, often display more robust tail behavior due to diversification.

---

## CAPM folder

The `CAPM` directory contains solutions to exercises and assignments from the Coursera course:

Optimization Methods in Asset Management  
https://www.coursera.org/learn/financial-engineering-optimizationmethods?specialization=financialengineering

These materials are included for educational purposes only and are clearly separated from the original project code.

---

## Methods and assumptions

- Mean–variance framework with sample estimates of expected returns and covariance
- Daily returns used consistently across optimization and risk metrics
- Sharpe ratio computed using a constant daily risk-free rate
- VaR and CVaR computed using a loss-based convention
- No explicit tail-risk constraints unless stated otherwise

