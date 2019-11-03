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
This may take a little while, as two stan models are compiled as part of the build. You may also see an error during installation whereby pip fails to build a wheel for `bpl` -- this isn't a problem. Installation will continue, and it will run `python setup.py install` instead, which should install the package fine.

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

## Statistical model

The statistical model behind `bpl` is a slight variation on the Dixon & Coles approach.
The likelihood is:

$$p(y_h, y_a) = \tau(y_h, y_a)\times \mathrm{Poisson}(y_h \, | \, a_h b_a \gamma) \times \mathrm{Poisson}(y_a \, | \, a_a b_h)$$

where $y_h$ and $y_a$ are the number of goals scored by the home team and the away team, respectively.
$a_i$ is the *attacking aptitude* of team $i$ and $b_i$ is the *defending aptitude* of team $j$.
$\gamma$ represents the home advantage, and $\tau$ is a correlation term that was introduced by Dixon and Coles to produce more realistic scorelines in low-scoring matches.
The model uses the following bivariate, hierarchical prior for $a$ and $b$

$$
\begin{bmatrix}
\log a_i \\
\log b_i
\end{bmatrix} \, \big | \, X_i\sim \mathcal{N}
\left(
\begin{bmatrix}
X_i . \beta_a \\
\mu_b + X_i . \beta_b
\end{bmatrix},\quad
\begin{bmatrix}
\sigma_a^2, \quad \rho \sigma_a \sigma_b \\
\rho \sigma_a \sigma_b, \quad \sigma_b^2
\end{bmatrix}    
\right).
$$
$X_i$ are a set of (optional) team-level covariates (these could be, for example, the attack and defence ratings of team $i$ on Fifa).
$\beta_{a, b}$ are coefficient vectors, and $\mu_b$ is an offset for the defence parameter.
$\rho$ encodes the correlation between $a$ and $b$, since teams that are strong at attacking also tend to be strong at defending as well.
The home advantage has a straightforward log-normal prior

$$
\gamma \sim \mathrm{LogNormal}(0, 1),
$$


Finally, the hyper-priors are

$$
\mu_b, \beta_a, \beta_b \sim \mathcal{N}(0, 1), \\
\sigma_a, \sigma_b \sim \mathcal{N}^+(0, 1), \\
u = (\rho + 1) / 2 \sim  \mathrm{Beta}(2, 4).
$$

