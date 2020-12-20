from dfit import get_distributions, DFit
import pytest


def test_get_distributions():
    assert "beta" in get_distributions("popular"), "beta not in popular distributions"
    # test blacklist
    assert all(i not in get_distributions() for i in ["rv_histogram", "rv_continuous"])
    assert len(get_distributions()) >= 80


class TestDFit:
    def test_distributions_input(self):
        DFit([1, 1], distributions="popular")
        DFit([1, 1], distributions="all")
        DFit([1, 1], distributions="beta")
        DFit([1, 1], distributions=["gamma"])
        with pytest.raises(ValueError):
            DFit([1, 1], distributions="abczyz")

    def test_min_max(self):
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

        df = DFit(data)
        assert df.xmin == 1
        assert df.xmax == 4

        df.xmin = 2
        df.xmax = 3
        assert df.xmin == 2
        assert df.xmax == 3

        # reset
        df.xmin = -1
        df.xmax = 1e100
        assert df.xmin == 1
        assert df.xmax == 4

    def test_plot(self, data_beta_1k):
        df = DFit(data_beta_1k, distributions=["gamma", "beta"], bins="auto")
        df.plot_hist()
        df.fit()
        df.plot_pdf()
        df.summary()
        assert "beta" in df.get_best()

    def test_dfit_basic(self):
        df = DFit([1, 1, 2, 2, 2, 3, 3, 3, 3], distributions="beta")
        df.plot_hist()
        df.fit()
        df.summary()
        assert "beta" in df.get_best()

    def test_fit(self, data_beta_1k):
        df = DFit(data_beta_1k, distributions="popular")
        df.fit()
        assert df.df_errors.loc["beta"].loc["aic"] > 90

        df = DFit(data_beta_1k, distributions="beta")
        df.fit()
        assert df.df_errors.loc["beta"].loc["aic"] > 90
