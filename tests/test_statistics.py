"""Tests for statistical analysis functions."""

import pytest

from qpso_research.statistics import wilcoxon_test, friedman_test, format_results_table


class TestWilcoxon:

    def test_identical_scores_not_significant(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = wilcoxon_test(scores, scores)
        assert not result.significant
        assert result.p_value == 1.0

    def test_different_scores_detected(self):
        # Clearly different distributions
        a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = wilcoxon_test(a, b)
        assert result.significant
        assert result.p_value < 0.05

    def test_result_has_correct_fields(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        result = wilcoxon_test(a, b)
        assert result.test_name == "Wilcoxon signed-rank"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)


class TestFriedman:

    def test_three_groups(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        c = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = friedman_test(a, b, c)
        assert result.test_name == "Friedman"
        assert isinstance(result.p_value, float)


class TestFormatTable:

    def test_produces_markdown(self):
        results = {
            "sphere_2d": {
                "PSO": {"mean": 1.0e-5, "std": 5.0e-6, "best": 1.0e-6},
                "QPSO": {"mean": 5.0e-6, "std": 2.0e-6, "best": 5.0e-7},
            }
        }
        table = format_results_table(results, ["sphere_2d"])
        assert "PSO" in table
        assert "QPSO" in table
        assert "|" in table
