import numpy as np
import pandas as pd

import warnings

from scipy.stats import poisson

from bpl.stan_models import simple_stan_model, prior_stan_model
from bpl.util import ModelNotFitError, ModelNotConvergedWarning, suppress_output


class BPLModel:
    """Class for fitting and manipulating models."""

    def __init__(self, data, X=None):
        """
        :param df: a pandas dataframe containing the data. Must have columns "home_team", "away_team", "home_goals",
            "away_goals" and "date".
        :param X: a pandas dataframe containing extra covariates to use in the prior for the team aptitudes.
            There should be a column "team", where each team in the league should only appear once. The remaining
            columns correspond to the features (which currently must be numerical).
        """
        self.model = prior_stan_model if X is not None else simple_stan_model
        self.X = X
        self.pandas_data = data
        self.stan_data = None
        self.team_indices = {
            team: i + 1 for i, team in enumerate(np.sort(data["home_team"].unique()))
        }
        self._is_fit = False
        self.a = None
        self.b = None
        self.gamma = None
        self.beta_a = None
        self.beta_b = None
        self.beta_b_0 = None

    def _pre_process_data(self, max_date=None):
        """Preprocess the data for stan."""
        df = self.pandas_data.replace(
            to_replace={"home_team": self.team_indices, "away_team": self.team_indices}
        )
        df.loc[:, "date"] = pd.to_datetime(df["date"])
        if max_date:
            df = df[df["date"] <= max_date]
            if len(df) == 0:
                raise ValueError("No games before this date.")
        stan_data = dict(
            nteam=df["home_team"].nunique(),
            nmatch=len(df),
            home_team=df["home_team"].values,
            away_team=df["away_team"].values,
            home_goals=df["home_goals"].values,
            away_goals=df["away_goals"].values,
        )
        if self.X is not None:
            stan_X = (
                self.X.replace(to_replace={"team": self.team_indices})
                .sort_values("team")
                .drop("team", axis=1)
                .astype(float)
                .values
            )
            stan_X = (
                0.5
                * (stan_X - stan_X[:, None, :].mean(axis=0))
                / stan_X[:, None, :].std(axis=0)
            )
            stan_data["X"] = stan_X
            stan_data["nfeat"] = stan_X.shape[1]
        return stan_data

    def fit(self, max_date=None, return_summary=False, **stan_kwargs):
        """
        Fit the model.

        Run Stan's NUTS sampler to draw samples from the posterior distribution of the model parameters given the
        data provided.

        :param max_date: fit the model to games that took place on or before this date.
        :param return_summary: if True, return a pandas dataframe with summary statistics from the MCMC sampling.
        :param stan_kwargs: extra keyword arguments that are passed to the StanModel (e.g., niter).
        :return: a stan fit object. If `return_summary` is True, then a pandas dataframe containing a summary of the
            sampling is also returned.
        """
        stan_data = self._pre_process_data(max_date)
        with suppress_output():
            fit = self.model.sampling(data=stan_data, **stan_kwargs)
        self.a = fit["a"]
        self.b = fit["b"]
        self.gamma = fit["gamma"]
        if self.X is not None:
            self.beta_a = fit["beta_a"]
            self.beta_b = fit["beta_b"]
            self.beta_b_0 = fit["beta_b_0"]
        self._is_fit = True

        s = fit.summary()
        summary = pd.DataFrame(
            s["summary"], columns=s["summary_colnames"], index=s["summary_rownames"]
        )
        if (summary["Rhat"] > 1.1).any():
            warnings.warn(
                "Sampling may not have converged - some scale reduction factors are > 1.1."
                " Try running again with more iterations.",
                ModelNotConvergedWarning,
            )
        if return_summary:
            return summary
        else:
            return

    def simulate_match(self, home_team, away_team):
        """
        Simulate a match.

        Given a home team and an away team, use the output from the model fit to produce a set of simulated matches
        for this match-up. Can be used for e.g. posterior predictive checks.

        :param home_team: name of the home team.
        :param away_team: name of the away team.
        :return: a pandas dataframe containing columns home_team and away_team, which are the results of
            the simulations.
        """
        if not self._is_fit:
            raise ModelNotFitError(
                "Model must be fit to data before forecasts can be made."
            )
        home_ind = self.team_indices[home_team] - 1
        away_ind = self.team_indices[away_team] - 1
        a_home, b_home = self.a[:, home_ind], self.b[:, home_ind]
        a_away, b_away = self.a[:, away_ind], self.b[:, away_ind]
        home_rate = a_home * b_away * self.gamma
        away_rate = a_away * b_home
        home_goals = np.random.poisson(home_rate)
        away_goals = np.random.poisson(away_rate)
        df = pd.DataFrame({home_team: home_goals, away_team: away_goals})
        return df

    def score_probability(self, home_team, away_team, home_goals, away_goals):
        """
        Compute the probability of a result.

        Given a home team and an away team, use the output from the model to produce the posterior predictive
        probability of the proposed result.

        :param home_team: name of the home team (must match a team name in self.data)
        :param away_team: name of the away team (must match a team name in self.data)
        :param home_goals: number of home goals.
        :param away_goals: number of away goals.
        :return: the probability of this result.
        """
        if not self._is_fit:
            raise ModelNotFitError(
                "Model must be fit to data before forecasts can be made."
            )
        home_ind = self.team_indices[home_team] - 1
        away_ind = self.team_indices[away_team] - 1
        a_home, b_home = self.a[:, home_ind], self.b[:, home_ind]
        a_away, b_away = self.a[:, away_ind], self.b[:, away_ind]
        home_rate = a_home * b_away * self.gamma
        away_rate = a_away * b_home
        home_probs = poisson.pmf(home_goals, home_rate)
        away_probs = poisson.pmf(away_goals, away_rate)
        return np.mean(home_probs * away_probs)

    def concede_n_probability(self, n, team, opponent, home=True):
        """
        Compute the probability that a team will concede n goals.

        Given a team and an opponent, calculate the probability that the team will
        concede n goals against this opponent.

        :param n: the number of goals.
        :param team: the name of the team.
        :param opponent: the name of the opponent.
        :param home: (optional) if True, then it is assumed that the team are
            playing at home.
        """
        if not self._is_fit:
            raise ModelNotFitError(
                "Model must be fit to data before forecasts can be made."
            )
        team_ind = self.team_indices[team] - 1
        oppo_ind = self.team_indices[opponent] - 1
        b_team, a_oppo = self.b[:, team_ind], self.a[:, oppo_ind]
        score_rate = b_team * a_oppo
        if not home:
            score_rate *= self.gamma
        goal_probs = poisson.pmf(n, score_rate)
        return np.mean(goal_probs)

    def score_n_probability(self, n, team, opponent, home=True):
        """
        Compute the probability that a team will score n goals.

        Given a team and an opponent, calculate the probability that the team will
        score n goals against this opponent.

        :param n: the number of goals.
        :param team: the name of the team.
        :param opponent: the name of the opponent.
        :param home: (optional) if True, then it is assumed that the team are
            playing at home.
        """
        if not self._is_fit:
            raise ModelNotFitError(
                "Model must be fit to data before forecasts can be made."
            )
        team_ind = self.team_indices[team] - 1
        oppo_ind = self.team_indices[opponent] - 1
        a_team, b_oppo = self.a[:, team_ind], self.b[:, oppo_ind]
        score_rate = a_team * b_oppo
        if home:
            score_rate *= self.gamma
        goal_probs = poisson.pmf(n, score_rate)
        return np.mean(goal_probs)

    def overall_probabilities(self, home_team, away_team):
        """
        Compute the probability of home win, away win and draw.

        Given a home team and an away team, use the output from the model to produce posterior predictive
        probabilities of the three possible overall results.

        :param home_team: name of the home team (must match a team name in self.data)
        :param away_team: name of the away team (must match a team name in self.data)
        :return: home_win, away_win, draw - a tuple of floats corresponding to the three result probabilities.
        """
        if not self._is_fit:
            raise ModelNotFitError(
                "Model must be fit to data before forecasts can be made."
            )
        max_goals = 15
        n_goals = np.arange(0, max_goals + 1)
        x, y = np.meshgrid(n_goals, n_goals, indexing="ij")
        prob = np.array(
            [
                self.score_probability(home_team, away_team, xi, yj)
                for xi in n_goals
                for yj in n_goals
            ]
        ).reshape(max_goals + 1, max_goals + 1)
        home_win = np.sum(prob[x > y])
        away_win = np.sum(prob[x < y])
        draw = np.sum(prob[x == y])
        return home_win, away_win, draw
