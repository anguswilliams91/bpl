import numpy as np
import pandas as pd

import os

from unittest import TestCase

from bpl.models import BPLModel
from bpl.util import (
    ModelNotConvergedWarning,
    ModelNotFitError,
    UnseenTeamError,
    check_fit,
)

TEST_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_data.csv"))
TEST_FEATS = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_feats.csv"))
FITTED_MODEL = BPLModel(TEST_DATA)
FITTED_MODEL.fit(iter=1000, seed=42)


class TestBPLModel(TestCase):
    def test_preprocess_data_nofeats(self):
        """Test that stan data dictionary has the correct keys when no features are passed."""
        model = BPLModel(TEST_DATA, X=None)
        stan_data = model._pre_process_data()
        self.assertTrue(
            set(stan_data.keys())
            == {"nteam", "nmatch", "home_team", "away_team", "home_goals", "away_goals"}
        )

    def test_preprocess_data_feats(self):
        """Test that the stan data dictionary has the correct keys when features are passed."""
        model = BPLModel(TEST_DATA, X=TEST_FEATS)
        stan_data = model._pre_process_data()
        self.assertTrue(
            set(stan_data.keys())
            == {
                "nteam",
                "nmatch",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "nfeat",
                "X",
            }
        )

    def test_preprocess_data_date(self):
        """Test that the correct date constraint is applied in preprocessing."""
        model = BPLModel(TEST_DATA, X=TEST_FEATS)
        stan_data = model._pre_process_data(max_date="2018-03-01")
        df = TEST_DATA.copy()
        df.loc[:, "date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= "2018-03-01"]
        n = len(df)
        self.assertEqual(stan_data["nmatch"], n)

    def test_preprocess_data_baddate(self):
        """Test that the correct error is raised if max_date is too low."""
        model = BPLModel(TEST_DATA, X=TEST_FEATS)
        self.assertRaises(ValueError, model._pre_process_data, max_date="1966-07-30")

    def test_preprocess_data_scaling_mean(self):
        """Test that the input features have been correctly scaled to zero mean."""
        model = BPLModel(TEST_DATA, X=TEST_FEATS)
        stan_data = model._pre_process_data()
        self.assertTrue(
            np.allclose(stan_data["X"].mean(axis=0), np.zeros(stan_data["X"].shape[1]))
        )

    def test_preprocess_data_scaling_mean(self):
        """Test that the input features have been correctly scaled to 0.5 standard deviation."""
        model = BPLModel(TEST_DATA, X=TEST_FEATS)
        stan_data = model._pre_process_data()
        self.assertTrue(
            np.allclose(
                stan_data["X"].std(axis=0), 0.5 * np.ones(stan_data["X"].shape[1])
            )
        )

    def test_fit_nofeats_run(self):
        """Test that the fit method runs."""
        model = BPLModel(TEST_DATA)
        model.fit(iter=100, seed=42)

    def test_fit_feats_run(self):
        """Test that the fit method runs when we pass features."""
        model = BPLModel(TEST_DATA, X=TEST_FEATS)
        model.fit(iter=100, seed=42)

    def test_fit_nofeats_summary(self):
        """Test that the summary is returned as a pandas dataframe when asked for."""
        model = BPLModel(TEST_DATA)
        summary = model.fit(return_summary=True, iter=100, seed=42)
        self.assertIsInstance(summary, pd.DataFrame)

    def test_modelnotfit(self):
        """Test that the ModelNotFit error is thrown where is should be."""

        class DummyModel:
            def __init__(self, is_fit):
                self._is_fit = is_fit

            @check_fit
            def foo(self):
                return True

        dummy_not_fit = DummyModel(False)
        dummy_fit = DummyModel(True)
        # check error thrown when _is_fit is False
        self.assertRaises(ModelNotFitError, dummy_not_fit.foo)
        # check error not thrown when _is_fit is True
        self.assertTrue(dummy_fit.foo())

    def test_simulate_match(self):
        df = FITTED_MODEL.simulate_match("Arsenal", "Man City")
        self.assertTrue(set(df.columns) == {"Arsenal", "Man City"})
        self.assertTrue(len(df) == FITTED_MODEL.a.shape[0])
        self.assertEqual(df["Arsenal"].isnull().sum(), 0.0)
        self.assertEqual(df["Man City"].isnull().sum(), 0.0)

    def test_score_probabilities(self):
        """Test that the score probabilities are between 0 and 1."""
        pr = FITTED_MODEL.score_probability("Arsenal", "Man City", 1, 0)
        self.assertTrue(0.0 <= pr <= 1.0)

    def test_overall_probabilities(self):
        """Test that the overall probabilities sum close to 1."""
        pr = FITTED_MODEL.overall_probabilities("Arsenal", "Man City")
        self.assertAlmostEqual(sum(pr), 1.0, places=5)

    def test_score_n_probabilities(self):
        """Test method that computes the probability of scoring n goals."""
        pr_home = FITTED_MODEL.score_n_probability(1, "Arsenal", "Man City", home=True)
        pr_away = FITTED_MODEL.score_n_probability(1, "Arsenal", "Man City", home=False)
        self.assertTrue((0.0 <= pr_home <= 1.0) and (0.0 <= pr_away <= 1.0))

    def test_concede_n_probabilities(self):
        """Test method that computes the probability of scoring n goals."""
        pr_home = FITTED_MODEL.concede_n_probability(
            1, "Arsenal", "Man City", home=True
        )
        pr_away = FITTED_MODEL.concede_n_probability(
            1, "Arsenal", "Man City", home=False
        )
        self.assertTrue((0.0 <= pr_home <= 1.0) and (0.0 <= pr_away <= 1.0))

    def test_predict_future_matches(self):
        """Test predict future matches"""
        df = pd.DataFrame(
            {
                "home_team": ["Arsenal", "Man City", "Tottenham"],
                "away_team": ["Man City", "Tottenham", "Arsenal"],
            }
        )
        df_pred = FITTED_MODEL.predict_future_matches(df)
        # check returned dataframe has correct columns
        self.assertSetEqual(
            set(df_pred.columns),
            {"home_team", "away_team", "pr_home", "pr_away", "pr_draw"},
        )
        # check probabilities sum to 1 across the columns
        self.assertTrue(
            np.allclose(
                df_pred["pr_home"] + df_pred["pr_away"] + df_pred["pr_draw"],
                [1.0] * len(df),
            )
        )

    def test_log_score(self):
        """Test log score calculation"""
        self.assertTrue(FITTED_MODEL.log_score() < 0.0)
        self.assertTrue(FITTED_MODEL.log_score(date_range=("2018-01-01", "2018-03-01")))
        df_mock = pd.DataFrame(
            {
                "date": ["2018-01-02"],
                "home_team": ["Man City"],
                "away_team": ["Arsenal"],
                "home_goals": [4.0],
                "away_goals": [1.0],
            }
        )
        self.assertTrue(FITTED_MODEL.log_score(df_mock))

    def test_add_new_team(self):
        """Test functionality for adding new teams"""
        model = BPLModel(data=TEST_DATA)
        model_X = BPLModel(data=TEST_DATA, X=TEST_FEATS)
        model.fit(iter=100)
        model_X.fit(iter=100)

        # check correct exception raised if unseen team is passed
        self.assertRaises(
            UnseenTeamError, model.overall_probabilities, "blah", "Man City"
        )
        self.assertRaises(
            UnseenTeamError, model.overall_probabilities, "Man City", "blah"
        )
        self.assertRaises(
            UnseenTeamError, model.overall_probabilities, "blah1", "blah2"
        )

        # add a new team and check shapes
        model.add_new_team("blah1")
        self.assertIn("blah1", model.team_indices.keys())
        self.assertEqual(model.a.shape[1], 21)
        self.assertEqual(model.b.shape[1], 21)

        # two new teams added with no extra prior information should be identical
        model.add_new_team("blah2")
        self.assertTupleEqual(
            model.overall_probabilities("blah1", "blah2"),
            model.overall_probabilities("blah2", "blah1"),
        )

        # if team already known is added, exception should be raised
        self.assertRaises(ValueError, model.add_new_team, "Man City")

        # test adding a new team to model with / without covariates
        model_X.add_new_team("blah1")
        model_X.add_new_team("blah2", X=np.array([70.0, 70.0, 70.0]))
