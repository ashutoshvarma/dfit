import pytest
import scipy.stats as stats
import dfit


def all_dists():
    # dists param were taken from scipy.stats official
    # documentaion examples
    # Total - 89
    return {
        "alpha": stats.alpha(a=3.57, loc=0.0, scale=1.0),
        "anglit": stats.anglit(loc=0.0, scale=1.0),
        "arcsine": stats.arcsine(loc=0.0, scale=1.0),
        "beta": stats.beta(a=2.31, b=0.627, loc=0.0, scale=1.0),
        "betaprime": stats.betaprime(a=5, b=6, loc=0.0, scale=1.0),
        "bradford": stats.bradford(c=0.299, loc=0.0, scale=1.0),
        "burr": stats.burr(c=10.5, d=4.3, loc=0.0, scale=1.0),
        "cauchy": stats.cauchy(loc=0.0, scale=1.0),
        "chi": stats.chi(df=78, loc=0.0, scale=1.0),
        "chi2": stats.chi2(df=55, loc=0.0, scale=1.0),
        "cosine": stats.cosine(loc=0.0, scale=1.0),
        "dgamma": stats.dgamma(a=1.1, loc=0.0, scale=1.0),
        "dweibull": stats.dweibull(c=2.07, loc=0.0, scale=1.0),
        "erlang": stats.erlang(a=2, loc=0.0, scale=1.0),
        "expon": stats.expon(loc=0.0, scale=1.0),
        "exponnorm": stats.exponnorm(K=1.5, loc=0.0, scale=1.0),
        "exponweib": stats.exponweib(a=2.89, c=1.95, loc=0.0, scale=1.0),
        "exponpow": stats.exponpow(b=2.7, loc=0.0, scale=1.0),
        "f": stats.f(dfn=29, dfd=18, loc=0.0, scale=1.0),
        "fatiguelife": stats.fatiguelife(c=29, loc=0.0, scale=1.0),
        "fisk": stats.fisk(c=3.09, loc=0.0, scale=1.0),
        "foldcauchy": stats.foldcauchy(c=4.72, loc=0.0, scale=1.0),
        "foldnorm": stats.foldnorm(c=1.95, loc=0.0, scale=1.0),
        # "frechet_r": stats.frechet_r(c=1.89, loc=0.0, scale=1.0),
        # "frechet_l": stats.frechet_l(c=3.63, loc=0.0, scale=1.0),
        "genlogistic": stats.genlogistic(c=0.412, loc=0.0, scale=1.0),
        "genpareto": stats.genpareto(c=0.1, loc=0.0, scale=1.0),
        "gennorm": stats.gennorm(beta=1.3, loc=0.0, scale=1.0),
        "genexpon": stats.genexpon(a=9.13, b=16.2, c=3.28, loc=0.0, scale=1.0),
        "genextreme": stats.genextreme(c=-0.1, loc=0.0, scale=1.0),
        "gausshyper": stats.gausshyper(
            a=13.8, b=3.12, c=2.51, z=5.18, loc=0.0, scale=1.0
        ),
        "gamma": stats.gamma(a=1.99, loc=0.0, scale=1.0),
        "gengamma": stats.gengamma(a=4.42, c=-3.12, loc=0.0, scale=1.0),
        "genhalflogistic": stats.genhalflogistic(c=0.773, loc=0.0, scale=1.0),
        "gilbrat": stats.gilbrat(loc=0.0, scale=1.0),
        "gompertz": stats.gompertz(c=0.947, loc=0.0, scale=1.0),
        "gumbel_r": stats.gumbel_r(loc=0.0, scale=1.0),
        "gumbel_l": stats.gumbel_l(loc=0.0, scale=1.0),
        "halfcauchy": stats.halfcauchy(loc=0.0, scale=1.0),
        "halflogistic": stats.halflogistic(loc=0.0, scale=1.0),
        "halfnorm": stats.halfnorm(loc=0.0, scale=1.0),
        "halfgennorm": stats.halfgennorm(beta=0.675, loc=0.0, scale=1.0),
        "hypsecant": stats.hypsecant(loc=0.0, scale=1.0),
        "invgamma": stats.invgamma(a=4.07, loc=0.0, scale=1.0),
        "invgauss": stats.invgauss(mu=0.145, loc=0.0, scale=1.0),
        "invweibull": stats.invweibull(c=10.6, loc=0.0, scale=1.0),
        "johnsonsb": stats.johnsonsb(a=4.32, b=3.18, loc=0.0, scale=1.0),
        "johnsonsu": stats.johnsonsu(a=2.55, b=2.25, loc=0.0, scale=1.0),
        "ksone": stats.ksone(n=1e03, loc=0.0, scale=1.0),
        "kstwobign": stats.kstwobign(loc=0.0, scale=1.0),
        "laplace": stats.laplace(loc=0.0, scale=1.0),
        "levy": stats.levy(loc=0.0, scale=1.0),
        "levy_l": stats.levy_l(loc=0.0, scale=1.0),
        "levy_stable": stats.levy_stable(alpha=0.357, beta=-0.675, loc=0.0, scale=1.0),
        "logistic": stats.logistic(loc=0.0, scale=1.0),
        "loggamma": stats.loggamma(c=0.414, loc=0.0, scale=1.0),
        "loglaplace": stats.loglaplace(c=3.25, loc=0.0, scale=1.0),
        "lognorm": stats.lognorm(s=0.954, loc=0.0, scale=1.0),
        "lomax": stats.lomax(c=1.88, loc=0.0, scale=1.0),
        "maxwell": stats.maxwell(loc=0.0, scale=1.0),
        "mielke": stats.mielke(k=10.4, s=3.6, loc=0.0, scale=1.0),
        "nakagami": stats.nakagami(nu=4.97, loc=0.0, scale=1.0),
        "ncx2": stats.ncx2(df=21, nc=1.06, loc=0.0, scale=1.0),
        "ncf": stats.ncf(dfn=27, dfd=27, nc=0.416, loc=0.0, scale=1.0),
        "nct": stats.nct(df=14, nc=0.24, loc=0.0, scale=1.0),
        "norm": stats.norm(loc=0.0, scale=1.0),
        "pareto": stats.pareto(b=2.62, loc=0.0, scale=1.0),
        "pearson3": stats.pearson3(skew=0.1, loc=0.0, scale=1.0),
        "powerlaw": stats.powerlaw(a=1.66, loc=0.0, scale=1.0),
        "powerlognorm": stats.powerlognorm(c=2.14, s=0.446, loc=0.0, scale=1.0),
        "powernorm": stats.powernorm(c=4.45, loc=0.0, scale=1.0),
        "rdist": stats.rdist(c=0.9, loc=0.0, scale=1.0),
        "reciprocal": stats.reciprocal(a=0.00623, b=1.01, loc=0.0, scale=1.0),
        "rayleigh": stats.rayleigh(loc=0.0, scale=1.0),
        "rice": stats.rice(b=0.775, loc=0.0, scale=1.0),
        "recipinvgauss": stats.recipinvgauss(mu=0.63, loc=0.0, scale=1.0),
        "semicircular": stats.semicircular(loc=0.0, scale=1.0),
        "t": stats.t(df=2.74, loc=0.0, scale=1.0),
        "triang": stats.triang(c=0.158, loc=0.0, scale=1.0),
        "truncexpon": stats.truncexpon(b=4.69, loc=0.0, scale=1.0),
        "truncnorm": stats.truncnorm(a=0.1, b=2, loc=0.0, scale=1.0),
        "tukeylambda": stats.tukeylambda(lam=3.13, loc=0.0, scale=1.0),
        "uniform": stats.uniform(loc=0.0, scale=1.0),
        "vonmises": stats.vonmises(kappa=3.99, loc=0.0, scale=1.0),
        "vonmises_line": stats.vonmises_line(kappa=3.99, loc=0.0, scale=1.0),
        "wald": stats.wald(loc=0.0, scale=1.0),
        "weibull_min": stats.weibull_min(c=1.79, loc=0.0, scale=1.0),
        "weibull_max": stats.weibull_max(c=2.87, loc=0.0, scale=1.0),
        "wrapcauchy": stats.wrapcauchy(c=0.0311, loc=0.0, scale=1.0),
    }


@pytest.fixture
def data_beta_1k():
    return stats.beta.rvs(2, 1.5, scale=2, size=int(1e4))


@pytest.fixture
def popular_data_1k():
    return {
        name: dist.rvs(size=1000)
        for name, dist in all_dists().items()
        if name in dfit.get_distributions("popular")
    }
