import numpy as np
import pandas as pd
import tqdm

import warnings

from scipy.stats import poisson

from bpl.plotting import plot_score_grid, has_matplotlib
from bpl.stan_models import simple_stan_model, prior_stan_model
from bpl.util import (
    ModelNotConvergedWarning,
    UnseenTeamError,
    suppress_output,
    check_fit,
)


class BPLModel:
    """Class for fitting and manipulating models."""

    def __init__(self, data, X=None):
        """
        :param df: a pandas dataframe containing the data. Must have columns "home_team", "away_team", "home_goals",
            "away_goals" and "date". Date should be a pandas Timestamp.
        :param X: a pandas dataframe containing extra covariates to use in the prior for the team aptitudes.
            There should be a column "team", where each team in the league should only appear once. The remaining
            columns correspond to the features (which currently must be numerical).
        """
        self.model = prior_stan_model if X is not None else simple_stan_model
        self.pandas_data = data
        self.stan_data = None
        teams = list(set(data["home_team"]).union(set(data["away_team"])))
        self.team_indices = {
            team: i + 1 for i, team in enumerate(teams)
        }
        if X is not None:
            X = X[X["team"].isin(teams)]
            if any(~pd.Series(teams).isin(X["team"])):
                missing_teams = pd.Series(teams)[~pd.Series(teams).isin(X["team"])]
                print(" ".join(missing_teams))
                raise ValueError(
                    "Teams in X must match teams in data. Teams missing from X: {}".format(
                        " ".join(missing_teams.values)
                    )
                )
        self.X = X
        self._is_fit = False
        self.a = None
        self.b = None
        self.gamma = None
        self.beta_a = None
        self.beta_b = None
        self.beta_b_0 = None
        self.sigma_a = None
        self.sigma_b = None
        self.mu_b = None

    def _pre_process_data(self, max_date=None):
        """Preprocess the data for stan."""
        df = self.pandas_data.replace(
            to_replace={"home_team": self.team_indices, "away_team": self.team_indices}
        )
        if max_date:
            df = df[df["date"] <= max_date]
            if len(df) == 0:
                raise ValueError("No games before this date.")
        stan_data = dict(
            nteam=len(set(df["home_team"]).union(set(df["away_team"]))),
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
        self.stan_data = stan_data
        with suppress_output():
            fit = self.model.sampling(data=stan_data, **stan_kwargs)
        self.a = fit["a"]
        self.b = fit["b"]
        self.gamma = fit["gamma"]
        self.sigma_a = fit["sigma_a"]
        self.sigma_b = fit["sigma_b"]
        if self.X is not None:
            self.beta_a = fit["beta_a"]
            self.beta_b = fit["beta_b"]
            self.beta_b_0 = fit["beta_b_0"]
        else:
            self.mu_b = fit["mu_b"]
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

    def _check_teams_present(self, team_1, team_2):
        # check if two teams are known to the model
        def _make_error_msg(x):
            return "Cannot find team(s) {}. Use the add_new_team method and try again.".format(
                x
            )

        expr_1 = team_1 not in self.team_indices.keys()
        expr_2 = team_2 not in self.team_indices.keys()
        if expr_1 and expr_2:
            msg = team_1 + " and " + team_2
            raise UnseenTeamError(_make_error_msg(msg))
        elif expr_1:
            raise UnseenTeamError(_make_error_msg(team_1))
        elif expr_2:
            raise UnseenTeamError(_make_error_msg(team_2))
        else:
            pass

    @check_fit
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
        self._check_teams_present(home_team, away_team)
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

    @check_fit
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
        self._check_teams_present(home_team, away_team)
        home_ind = self.team_indices[home_team] - 1
        away_ind = self.team_indices[away_team] - 1
        a_home, b_home = self.a[:, home_ind], self.b[:, home_ind]
        a_away, b_away = self.a[:, away_ind], self.b[:, away_ind]
        home_rate = a_home * b_away * self.gamma
        away_rate = a_away * b_home
        home_probs = poisson.pmf(home_goals, home_rate)
        away_probs = poisson.pmf(away_goals, away_rate)
        return np.mean(home_probs * away_probs)

    @check_fit
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
        self._check_teams_present(team, opponent)
        team_ind = self.team_indices[team] - 1
        oppo_ind = self.team_indices[opponent] - 1
        b_team, a_oppo = self.b[:, team_ind], self.a[:, oppo_ind]
        score_rate = b_team * a_oppo
        if not home:
            score_rate *= self.gamma
        goal_probs = poisson.pmf(n, score_rate)
        return np.mean(goal_probs)

    @check_fit
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
        self._check_teams_present(team, opponent)
        team_ind = self.team_indices[team] - 1
        oppo_ind = self.team_indices[opponent] - 1
        a_team, b_oppo = self.a[:, team_ind], self.b[:, oppo_ind]
        score_rate = a_team * b_oppo
        if home:
            score_rate *= self.gamma
        goal_probs = poisson.pmf(n, score_rate)
        return np.mean(goal_probs)

    @check_fit
    def _make_score_grid(self, home_team, away_team, max_goals=15):
        # produce an "ij" indexed grid containing probabilities for various scorelines
        n_goals = np.arange(0, max_goals + 1)
        x, y = np.meshgrid(n_goals, n_goals, indexing="ij")
        prob = np.array(
            [
                self.score_probability(home_team, away_team, xi, yj)
                for xi in n_goals
                for yj in n_goals
            ]
        ).reshape(max_goals + 1, max_goals + 1)
        return prob, x, y

    def overall_probabilities(self, home_team, away_team):
        """
        Compute the probability of home win, away win and draw.

        Given a home team and an away team, use the output from the model to produce posterior predictive
        probabilities of the three possible overall results.

        :param home_team: name of the home team (must match a team name in self.data)
        :param away_team: name of the away team (must match a team name in self.data)
        :return: home_win, away_win, draw - a tuple of floats corresponding to the three result probabilities.
        """
        prob, x, y = self._make_score_grid(home_team, away_team)
        home_win = np.sum(prob[x > y])
        away_win = np.sum(prob[x < y])
        draw = np.sum(prob[x == y])
        return home_win, away_win, draw

    def predict_future_matches(self, future_matches):
        """
        Produce a pandas dataframe with predicted probabilities for a set of upcoming matches.

        :param future_matches: a pandas dataframe with columns "home_team", "away_team"
            Other columns may also be present, and these will be preserved in the output
        :return: a pandas dataframe with columns "home_team", "away_team", "pr_home", "pr_away" and "pr_draw",
            plus any extra columns in `future_matches`
        """
        df = future_matches.copy()
        probs = [
            self.overall_probabilities(home_team, away_team)
            for (home_team, away_team) in zip(df["home_team"], df["away_team"])
        ]
        df["pr_home"] = [p[0] for p in probs]
        df["pr_away"] = [p[1] for p in probs]
        df["pr_draw"] = [p[2] for p in probs]
        return df

    def plot_score_probabilities(self, home_team, away_team):
        """
        Visualise the probability of different scorelines.

        Produce a heatmap of various scorelines between the given home and away teams.
        A cross is drawn at the most likely scoreline.

        :param home_team: the home team
        :param away_team: the away team
        :return: matplotlib.figure.Figure object.
        """
        prob, _, __ = self._make_score_grid(home_team, away_team, max_goals=7)
        return plot_score_grid(prob, home_team, away_team)

    @check_fit
    def _log_score(self, df):
        # calculate the log score of the model on some matches
        pr_result = df.apply(
            lambda row: self.score_probability(
                row["home_team"], row["away_team"], row["home_goals"], row["away_goals"]
            ),
            axis=1,
        )
        return np.log(pr_result).sum() / len(pr_result)

    @check_fit
    def log_score(self, df=None, date_range=None):
        """
        Evaluate the log score of the model on a dataset.

        For each match, the log probability of the scoreline is computed using the predictive distribution of the
        model. The average log score is then calculated.


        :param df: (optional) a pandas dataframe containing columns "date", "home_team", "away_team",
            "home_goals" and "away_goals". If none is provided, the training data are used.
        :param date_range: a tuple of two datetimes. The first is the earliest date to consider, the second is
            the latest date to consider. This range is then applied to `df`.
        :return: the log score.
        """
        if df is None:
            df = self.pandas_data.copy()
        if date_range is not None:
            df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]
        return self._log_score(df)

    def cross_val_score(self):
        """
        Cross validation score for the model.

        Evaluate the log score by stepping through each date in the dataset and computing the log score for this date
        when the model is trained using data from prior to this date. The sum of the log score for all dates is
        returned. Note that this will be slow, because it requires fitting the model many times.

        :return: the cross validation log score
        """
        dates = self.pandas_data["date"].sort_values().unique()

        # find the first date where we have seen all of the teams
        seen_all_teams = False
        i = 0
        min_date = dates[0]
        while not seen_all_teams:
            df_sub = self.pandas_data[self.pandas_data["date"] <= min_date]
            if set(list(df_sub["home_team"]) + list(df_sub["away_team"])) == set(
                self.team_indices.keys()
            ):
                seen_all_teams = True
            else:
                i += 1
                min_date = dates[i]

        # iterate through each unique game day and store the log score
        cv_score = 0
        fit_dates = dates[dates >= min_date]
        for i, date in tqdm.tqdm(enumerate(fit_dates[:-1]), total=len(fit_dates[:-1])):
            self.fit(max_date=fit_dates[i])
            cv_score += self._log_score(
                self.pandas_data[self.pandas_data["date"] == fit_dates[i + 1]]
            )

        return cv_score / len(fit_dates[:-1])

    def add_new_team(self, team_name, X=None, seed=42):
        """Add an unseen team to the model by sampling from the prior.

        Add a new team to the model without explicitly possessing data for the team. This is done by sampling
        from the prior for the abilities a and b (integrating over the hyper-parameters). If the model has been
        fitted using extra covariates for the teams, these can optionally be provided to produce better forecasts.

        :param team_name: the name of the team to add.
        :param X: (optional) the team-level covariates for the new team.
        :param seed: (optional) the random seed to use when generating the samples from the prior.
        """
        if team_name in self.team_indices.keys():
            raise ValueError("Team {} already known to model.".format(team_name))

        np.random.seed(seed)
        if self.beta_a is not None:
            if X is None:
                warnings.warn(
                    "You haven't provided features for {}."
                    " Assuming X is the average of known teams."
                    " For better forecasts, provide X.".format(team_name)
                )
                X = np.zeros(self.stan_data["nfeat"])
            else:
                stan_X = (
                    self.X.replace(to_replace={"team": self.team_indices})
                    .sort_values("team")
                    .drop("team", axis=1)
                    .astype(float)
                    .values
                )
                X = (
                    0.5
                    * (X - stan_X[:, None, :].mean(axis=0))
                    / stan_X[:, None, :].std(axis=0)
                )
            mu_a = np.dot(self.beta_a, X.ravel())
            mu_b = self.beta_b_0 + np.dot(self.beta_b, X.ravel())
        else:
            mu_a = 0.0
            mu_b = self.mu_b

        log_a_tilde = np.random.normal(loc=0.0, scale=1.0, size=len(self.sigma_a))
        log_b_tilde = np.random.normal(loc=0.0, scale=1.0, size=len(self.sigma_a))
        a = np.exp(mu_a + log_a_tilde * self.sigma_a)
        b = np.exp(mu_b + log_b_tilde * self.sigma_b)

        new_index = max(self.team_indices.values()) + 1
        self.team_indices[team_name] = new_index
        self.a = np.concatenate((self.a, a[:, None]), axis=1)
        self.b = np.concatenate((self.b, b[:, None]), axis=1)
