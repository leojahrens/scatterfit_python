# imports
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
%matplotlib
import numpy as np
import pandas as pd
import scatterfit as sf

# Create dummy data or load your 'cpds' DataFrame
np.random.seed(234)
n = 500
data = pd.DataFrame({
    'x1': np.random.normal(size=n),
    'x2': np.random.normal(size=n),
    'x3': np.random.normal(size=n),
    'z': np.random.binomial(n=1, p=.2, size=n),
})
data['y'] = data['x1']*.5 + data['x2']*data['x2']*.2 + data['x3']*.7 + np.random.normal()
data["x1_help"] = ((data["x1"]-np.min(data["x1"])) / (np.max(data["x1"])-np.min(data["x1"]))) * .5
data['y_dum'] = np.random.binomial(n=1, p=data["x1_help"])

# simple linear fit
scatterfit(data, y_var='y', x_var='x1')

# change the fit line
scatterfit(data, y_var='y', x_var='x2', fit="lpoly")

# with data binning
scatterfit(data, y_var='y', x_var='x1', binned=True)

# binary y variable
scatterfit(data, y_var='y_dum', x_var='x1', binned=True)

# uniform bins
scatterfit(data, y_var='y', x_var='x1', binned=True, uni_bins=40)

# with weighted markers
scatterfit(data, y_var='y', x_var='x1', binned=True, uni_bins=40, mweighted=True)

# separate fits for by-dimension
scatterfit(data, y_var='y', x_var='x1', by_var="z")

# residualized data / control variables
scatterfit(data, y_var='y', x_var='x1', controls=["x2"], fcontrols=["z"])

# regression parameters in plot
scatterfit(data, y_var='y', x_var='x1', regparameters=["coef","sig","nobs","pval"])

# distribution of x variable
scatterfit(data, y_var='y', x_var='x1', xdistribution="auto")

# create a figure with multiple subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5)) # Adjust figsize as needed
for i, xvar in enumerate(["x1","x2","x3"]):
    scatterfit(data=data, y_var='y', x_var=xvar, ax=axes[i])
plt.tight_layout()
plt.show()









