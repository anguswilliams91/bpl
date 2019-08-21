# bpl

[![Build Status](https://travis-ci.org/anguswilliams91/bpl.svg?branch=master)](https://travis-ci.org/anguswilliams91/bpl)
[![codecov](https://codecov.io/gh/anguswilliams91/bpl/branch/master/graph/badge.svg)](https://codecov.io/gh/anguswilliams91/bpl)

`bpl` is a python 3 library for fitting Bayesian versions of the Dixon \& Coles (1997) model to data.
It uses the `stan` library to fit models to data.  

 ## Installation

You will need a working C++ compiler.
If you are using anaconda, you can install gcc with  

```bash
conda install gcc
``` 

You can then install with `pip`:

```bash
pip install bpl
```

## Usage

`bpl` provides a class `BPLModel` that can be used to forecast the outcome of football matches.
Data should be provided to the model as a `pandas` dataframe, with columns `home_team`, `away_team`, `home_goals` and `away_goals`.
You can also optionally provide a set of numerical covariates for each team (e.g. their ratings on FIFA) and these will be used in the fit.
Example usage:

```python
import bpl
import pandas as pd

df_train = pd.read_csv("<path-to-training-data>")
df_X = pd.read_csv("<path-to-team-level-covariates>")
forecaster = bpl.BPLModel(data=df_train, X=df_X)
forecaster.fit(seed=42)

# calculate the probability that team 1 beats team 2 3-0 at home:
forecaster.score_probability("Team 1", "Team 2", 3, 0)

# calculate the probabilities of a home win, an away win and a draw:
forecaster.overall_probabilities("Team 1", "Team 2")

# compute home win, away win and draw probabilities for a collection of matches:
df_test = pd.read_csv("<path-to-test-data>") # must have columns "home_team" and "away_team"
forecaster.predict_future_matches(df_test)

# add a new, previously unseen team to the model by sampling from the prior
X_3 = np.array([0.1, -0.5, 3.0]) # the covariates for the new team
forecaster.add_new_team("Team 3", X=X_3, seed=43)
```