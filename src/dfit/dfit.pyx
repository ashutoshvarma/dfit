# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++
# cython: language_level = 3

cimport cython
cimport numpy as cnp

cdef object logging
import logging

cdef object np
import numpy as np

cdef object sp
import scipy as sp

cdef object sp_stats
import scipy.stats as sp_stats

cdef object kl_div
from scipy.stats import entropy as kl_div

cdef pd
import pandas as pd

cdef plt
import matplotlib.pyplot as plt

cdef delayed
cdef Parallel
from joblib import delayed, Parallel




__all__ = [
    "DFit",
    "get_distributions"
]

logger = logging.getLogger(__name__)

include "interrupt.pxi"

# UTILITY
cdef _scipy_dists():
    dists = {}
    for attr_str in sp_stats.__all__:
        attr = getattr(sp_stats, attr_str)
        if getattr(attr, "fit", None):
            dists[attr_str] = attr
    return dists

cdef:
    dict _ALL_DISTS = _scipy_dists()
    list ALL_DISTRIBUTIONS = list(_ALL_DISTS.keys())
    list POPULAR_DISTRIBUTIONS = [
        d
        for d in [
            "norm",
            "lognorm",
            "expon",
            "gamma",
            "beta",
            "uniform",
            "pareto",
            "dweibull",
            "t",
            "genextreme",
        ]
        if d in ALL_DISTRIBUTIONS
    ]



def get_distributions(str dists="all"):
    if dists == "popular":
        return POPULAR_DISTRIBUTIONS
    else:
        return ALL_DISTRIBUTIONS

cdef class DFit:
    cdef:
        cnp.ndarray _raw_data
        cnp.ndarray _trim_data
        cnp.ndarray _pdf
        cnp.ndarray _x
        set distributions
        double _xmin
        double _xmax
        dict _sq_errors
        dict _aic
        dict _bic
        dict _kl
        # public properties
        public dict fitted_param
        public dict fitted_pdf
        public object timeout
        public object bins
        public object df_errors

    def __cinit__(
            self,
            data,
            tuple bound = (None, None),
            bins=50,
            distributions="popular",
            timeout=30
        ):
        self._raw_data = np.array(data)
        # self._raw_data = np.array(data, dtype=np.float64)
        # TODO: check for the case when bins are list, cause if bins are
        #       list of ints then np.histogram will return x/bins as np.int64
        #       but self._x is of type np.float64
        self.bins = bins
        self.timeout = timeout

        self._xmin = self._raw_data.min() if bound[0] == None else bound[0]
        self._xmax = self._raw_data.max() if bound[1] == None else bound[1]
        
        if distributions == "all":
            self.distributions = set(ALL_DISTRIBUTIONS)
        elif distributions == "popular":
            self.distributions = set(POPULAR_DISTRIBUTIONS)
        elif isinstance(distributions, str):
            if distributions not in ALL_DISTRIBUTIONS:
                raise Exception(f"Unknown distribution {distributions}")
            self.distributions = set([distributions])
        else:
            distributions = set(distributions)
            if not distributions.issubset(ALL_DISTRIBUTIONS):
                raise Exception(
                    f"Unknown distributions {distributions - set(ALL_DISTRIBUTIONS)}"
                )
            self.distributions = distributions

        self._init()
        self._update_trim_data()
        self._update_pdf()

    cdef _init(self):
        self._sq_errors = {}
        self._aic = {}
        self._bic = {}
        self._kl = {}
        self.fitted_param = {}
        self.fitted_pdf = {}

    cdef _update_trim_data(self):
        self._trim_data = self._raw_data[
            np.logical_and(
                self._raw_data >= self._xmin, self._raw_data <= self._xmax
            )
        ]

    cdef _update_pdf(self):
        self._pdf, x = np.histogram(self._trim_data, bins=self.bins, density=True)
        self._x = (x + np.roll(x, -1))[:-1] / 2.0


    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @xmin.setter
    def xmin(self, double value):
        cdef:
            double _raw_min = self._raw_data.min()
        if value <= _raw_min:
            value = _raw_min
        self._xmin = value
        self._update_trim_data()
        self._update_pdf()

    @xmax.setter
    def xmax(self, double value):
        cdef:
            double _raw_max = self._raw_data.max()
        if value <= _raw_max:
            value = _raw_max
        self._xmax = value
        self._update_trim_data()
        self._update_pdf()


    def _fit_distribution(self, str distribution):
        cdef:
            tuple param
            cnp.ndarray pdf_fitted

        cdef object warnings
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        dist = _ALL_DISTS[distribution]
        try:
            param = _timed_run(dist.fit, args=(self._trim_data, ), timeout=self.timeout)

            # assuming the `fit` return param in same order as in pdf
            fitted_pdf = dist.pdf(self._x, *param)

            self.fitted_param[distribution] = param[:]
            self.fitted_pdf[distribution] = fitted_pdf

            # square error
            sq_error = np.sum(
                (fitted_pdf - self._pdf) ** 2
            )

            # AIC & BIC
            log_likelihood = np.sum(dist.logpdf(self._x, *param))
            k = len(param)
            n = len(self._trim_data)
            aic = (2 * k) - (2 * log_likelihood)
            bic = n * np.log(sq_error / n) + k * np.log(n)

            # kullback leibler divergence
            # if self._pdf has some zero values it will make kl_div inf
            # as if p=0 then 1 * np.log(q/p) will be inf
            kl = kl_div(fitted_pdf, self._pdf)

            self._sq_errors[distribution] = sq_error
            self._aic[distribution] = aic
            self._bic[distribution] = bic
            self._kl[distribution] = kl
            # print(f"Fitted {distribution} with sq_error={sq_error}")
        except Exception as ex:
            self._sq_errors[distribution] = np.inf
            self._aic[distribution] = np.inf
            self._bic[distribution] = np.inf
            self._kl[distribution] = np.inf
            logging.warning(f"FAILED to fit {distribution} - {str(ex)}")


    def plot_hist(self):
        plt.hist(self._trim_data, bins=self.bins, density=True)
        plt.grid(True)

    def fit(self):
        jobs = (delayed(self._fit_distribution)(dist) for dist in self.distributions)
        pool = Parallel(n_jobs=-1, backend='threading')
        _ = pool(jobs)
        self.df_errors = pd.DataFrame(
            {
                'ss_error': self._sq_errors,
                'aic': self._aic,
                'bic': self._bic,
                'kl_div': self._kl
            }
        )

    def plot_pdf(self, distributions=None, Py_ssize_t n=5, lw=2, gof_metric="ss_error"):
        cdef:
            Py_ssize_t ld = len(self.distributions)

        assert n > 0
        if n > ld:
            n = ld

        if isinstance(distributions, list):
            for d in distributions:
                plt.plot(self._x, self.fitted_pdf[d], lw=lw, label=d)
        elif distributions:
            plt.plot(self._x, self.fitted_pdf[distributions], lw=lw, label=distributions)
        else:
            distributions = self.df_errors.sort_values(by=gof_metric).index[0:n]

        for d in distributions:
            if d in self.fitted_pdf.keys():
                plt.plot(
                    self._x, self.fitted_pdf[d], lw=lw, label=d
                )
            else:
                logger.warning(f"{d} was not fitted. Ignoring the distribution.")
        plt.grid(True)
        plt.legend()

    def get_best(self, gof_metric="ss_error"):
        cdef:
            str d = self.df_errors.sort_values(gof_metric).iloc[0].name
        return {d: self.fitted_param[d]}

    def summary(self, n=5, lw=2, plot=True, gof_metric="ss_error"):
        if plot:
            plt.clf()
            self.plot_hist()
            self.plot_pdf(n=n, lw=lw, gof_metric=gof_metric)

        n = min(n, len(self.distributions))
        dists = self.df_errors.sort_values(by=gof_metric).index[0:n]
        return self.df_errors.loc[dists]