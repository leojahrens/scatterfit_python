# scatterfit (Python version)

This Python package produces a wide range of scatter plots with overlaid fit lines. 

*   Visualize relationships between two continuous variables (`y` vs `x`).
*   Create standard or *binned* scatter plots (quantile, uniform, discrete value bins).
*   Overlay various fit lines (linear, quadratic, cubic, LOWESS/local polynomial) with optional confidence intervals.
*   Handle binary dependent variables using appropriate models (Logit, Probit, LPM) for fit lines.
*   Group plots by a categorical variable (`by_var`).
*   Control for covariates (`controls`, `fcontrols`) by plotting residualized relationships.
*   Display regression parameters directly on the plot.
*   Add a kernel density plot of the independent variable (`xdistribution`).
*   Customize appearance using Matplotlib/Seaborn themes and palettes.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/leojahrens/scatterfit_python.git
```

## How to use

Examples based on simulated data are available in the repository.