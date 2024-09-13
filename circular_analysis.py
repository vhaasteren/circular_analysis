import os

# For JAX/NumPyro
os.environ['JAX_PLATFORM_NAME']='cpu'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as sstats
import scipy.linalg as sl

import seaborn as sns

import json
from scipy import optimize as sopt

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def phitheta_to_psrpos(phi, theta):
    return np.array([np.cos(phi)*np.sin(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(theta)]).T

def generate_pulsar_positions(npsrs):
    """Generate uniformly distributed pulsar positions""" 
    phi = np.random.rand(npsrs) * 2 * np.pi
    theta = np.arccos(np.random.rand(npsrs) * 2 - 1)

    return phi, theta, phitheta_to_psrpos(phi, theta)

def hdfunc(gamma):
    cosgamma = np.clip(np.cos(gamma), -1.0, 1.0)
    xp = 0.5 * (1 - cosgamma)

    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.seterr(**old_settings)

    return logxp - 0.25 * xp + 0.5

def hdcorrmat(psrpos, psrTerm=True):
    """The definition as used in PTA literature"""

    cosgamma = np.clip(np.dot(psrpos, psrpos.T), -1, 1)
    npsrs = len(cosgamma)

    xp = 0.5 * (1 - cosgamma)

    # The settings make numpy ignore warnings due to numerical precision
    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old_settings)

    if psrTerm:
        coeff = 1.0
    else:
        coeff = 0.0
    hdmat = logxp - 0.25 * xp + 0.5 + coeff * 0.5 * np.diag(np.ones(npsrs))

    return hdmat

def get_correlations(x):
    """Given the positions, and the data, return correlations"""
    return np.dot(x, x.T) / x.shape[1]

def get_separation_angle(psrpos):
    return np.arccos(np.clip(np.dot(psrpos, psrpos.T), -1, 1))

def generate_data_fixh(
                  npsrs=36,
                  nobs=10,
                  correlations="hd",
                  log10_Arn_prior_mu=1.5,
                  log10_Arn_prior_sigma=1.0,
                  log10_hc_prior_mu=1.0,
                  prior_dist=sstats.norm,
                  **kwargs):

    # Create the data generation prior distributions
    log10_Arn_prior = prior_dist(loc=log10_Arn_prior_mu, scale=log10_Arn_prior_sigma)

    # The true values, drawn from the data generation priors
    log10_Arn_true = log10_Arn_prior.rvs(size=npsrs)
    log10_hc_true = log10_hc_prior_mu

    # The correlation matrix for the common signal
    phi, theta, psrpos = generate_pulsar_positions(npsrs)
    corrmat = hdcorrmat(psrpos, psrTerm=True) if correlations=="hd" else np.identity(npsrs)

    C_true = np.diag(10**log10_Arn_true) + corrmat * 10**log10_hc_true
    cholesky = sl.cholesky(C_true, lower=True)

    # The realization of data (npsrs x nobs)
    dt = np.dot(cholesky, np.random.randn(npsrs, nobs))

    return phi, theta, dt, log10_Arn_true, log10_hc_true, log10_Arn_prior_mu, log10_Arn_prior_sigma

def jax_log_likelihood_mv(log10_Arn, log10_hc, log10_Awn, corrmat, dt):
    C = jnp.diag(10**log10_Arn + 10**log10_Awn) + corrmat * 10**log10_hc
    cf = jsl.cho_factor(C, lower=True)

    logdet = jnp.sum(jnp.log(jnp.diag(cf[0])))
    xi2 = jnp.sum(dt * jsl.cho_solve(cf, dt), axis=0)

    return jnp.sum(-0.5*xi2 - logdet - np.log(2*np.pi))

def jax_log_likelihood_curn(log10_Arn, log10_hc, log10_Awn, dt):
    C = 10**log10_Arn + 10**log10_Awn + 10**log10_hc

    logdet = 0.5*jnp.sum(jnp.log(C))
    xi2 = jnp.sum(dt ** 2 / C[:,None], axis=0)

    return jnp.sum(-0.5*xi2 - logdet - np.log(2*np.pi))


def get_model(dt, corrmat, model='ER', log10_Awn=1.5, log10_Arn=None, prior_type="Hierarchical", corr_model="curn", log10_Arn_prior_mu=None, log10_Arn_prior_sigma=None, log10_hc_prior_mu=None, log10_hc_prior_sigma=None, bf_select=None):
    if bf_select is None and model.lower() == "data":
        raise ValueError("Require a bf_select for Data model")

    elif bf_select is None or model.lower() == "er":
        bf_select = np.ones(len(corrmat), dtype=bool)

    n_irn = np.sum(bf_select)
    npsrs = len(corrmat)

    def np_model():
        if prior_type == "Physical":
            if log10_Arn_prior_mu is not None and log10_Arn_prior_sigma is not None:
                # Physical priors (from data generation)
                log10_Arn = numpyro.sample('log10_Arn', dist.Normal(loc=log10_Arn_prior_mu, scale=log10_Arn_prior_sigma).expand([n_irn]))
                log10_hc = numpyro.sample('log10_hc', dist.Normal(loc=log10_hc_prior_mu, scale=log10_hc_prior_sigma))

            else:
                log10_Arn = numpyro.sample('log10_Arn', dist.Normal(loc=0, scale=10).expand([n_irn]))
                log10_hc = numpyro.sample("log10_hc", dist.Normal(loc=0, scale=5))

        elif prior_type == "Uniform":
            # PTA Noise & 'CURN'/'HD" priors (uniform priors)
            Arn = numpyro.sample("Arn", dist.Uniform(0, 10**9.0).expand([n_irn]))
            hc = numpyro.sample("hc", dist.Uniform(0, 10**3.0))

            log10_Arn = numpyro.deterministic("log10_Arn", jnp.log10(Arn))
            log10_hc = numpyro.deterministic("log10_hc", jnp.log10(hc))

        elif prior_type == "Log-Uniform":
            # PTA Noise & 'CURN'/'HD' priors (log-uniform priors)
            log10_Arn = numpyro.sample("log10_Arn", dist.Uniform(-3.0, 5.0).expand([n_irn]))
            log10_hc = numpyro.sample("log10_hc", dist.Uniform(-3.0, 5.0))

        elif prior_type == "Hierarchical":
            # PTA Noise & CURN'/'HD' priors
            log10_Arn_mean = numpyro.sample("log10_Arn_mean", dist.Normal(loc=1., scale=6.0))
            log10_Arn_sigma = numpyro.sample("log10_Arn_sigma", dist.Uniform(0.05, 4.1))

            # De-centered parameters
            log10_Arn_norm = numpyro.sample("log10_Arn_norm", dist.Normal(loc=0, scale=1).expand([n_irn]))
            log10_Arn = numpyro.deterministic("log10_Arn", log10_Arn_mean + log10_Arn_norm * log10_Arn_sigma)

            log10_hc = numpyro.sample("log10_hc", dist.Uniform(-3.0, 5.0))

        else:
            raise NotImplementedError(f"Prior type {prior_type} not implemented")

        log10_Arn_eff = -3 * jnp.ones(npsrs)
        log10_Arn_eff = log10_Arn_eff.at[bf_select].set(log10_Arn)


        if corr_model=="hd":
            numpyro.factor("log_likelihood", jax_log_likelihood_mv(log10_Arn_eff, log10_hc, log10_Awn, corrmat, dt))

        elif corr_model=="curn":
            N = 10**log10_Arn + 10**log10_hc
            numpyro.factor("log_likelihood", jax_log_likelihood_curn(log10_Arn_eff, log10_hc, log10_Awn, dt))

        else:
            raise NotImplementedError("Don't know this type of corr_model")

    return np_model

def get_indices(npsrs, method='all'):
    """
    parameters
    ----------
    :param method:      Can be 'all', 'auto', or 'cross'
    """

    if method=='all':
        # Both auto- and cross-correlations
        inds_first, inds_second = np.tril_indices(npsrs)

    elif method=='cross':
        # Only cross-correlations
        inds_first, inds_second = np.tril_indices(npsrs, -1)

    elif method=='auto':
        # Only cross-correlations
        inds_first, inds_second = np.diag_indices(npsrs)

    else:
        raise NotImplementedError("Can only do all-, cross-, and auto-correlations")

    return inds_first, inds_second

def build_bigC_mu(orf, log10_hc, log10_Arn, method='all'):
    """The Allen & Romano 'bigC'

    parameters
    ----------
    :param orf:         The GWB overlap reduction function matrix
                        For isotropic unpolarized, this is the H&D matrix
    :param log10_hc:    The log10 of hc
    :param log10_Arn:   The log10 of pulsar noise (vector)
    :param method:      Can be 'all', 'auto', or 'cross'
    """
    # The full signal and noise covariances
    hmu = orf * 10**log10_hc
    noise = np.diag(10**log10_Arn)
    rhob = hmu + noise

    inds_first, inds_second = get_indices(len(orf), method=method)

    # a/b label the rows of bigC. c/d label the columns of bigC
    a = inds_first[:,None]
    b = inds_second[:,None]
    c = inds_first[None,:]
    d = inds_second[None,:]

    bigC = rhob[(a,c)]*rhob[(b,d)] + rhob[(a,d)]*rhob[(b,c)]
    mu = orf[(inds_first, inds_second)]

    return bigC, mu

def correlate_data(dt, log10_Arn, method='all'):
    """dt is (npsrs x nobs)"""
    inds_first, inds_second = get_indices(len(dt), method=method)

    # We have multiple observations per pulsar.
    # Use correlation (outer product) only in row dimension
    rho = np.einsum('ij,kj->ikj', dt, dt)
    noise = np.diag(10**log10_Arn)[:,:,None]

    # Only use the triangular indices corresponding to 'method'
    return rho[inds_first, inds_second, :]

def order_by_separation(mu, rho, bigC, psrpos, method='all'):
    """Order by separation"""
    ia, ib = get_indices(len(psrpos), method=method)
    gamma = np.arccos(np.clip(np.dot(psrpos,psrpos.T), -1, 1))[(ia,ib)]
    isort = np.argsort(gamma)

    if len(gamma)!=len(mu):
        raise ValueError(f"Method {method} doesn't yield the correct number of correlation")

    mu = mu[isort]
    bigC = bigC[np.ix_(isort, isort)]
    gamma = gamma[isort]
    rho = rho[isort,:]

    return mu, rho, bigC, gamma

def subselect_by_bin(mu, rho, bigC, gamma, nbins=15):
    counts, edges = np.histogram(gamma, bins=nbins)

    bin_data = {}

    for ii in range(len(counts)):
        left,right = edges[ii:ii+2]
        inds = np.logical_and(gamma>=left,gamma<right)

        bin_data[ii] = {
            'left': left,
            'right': right,
            'bin': np.mean(edges[ii:ii+2]),
            'bigC': bigC[np.ix_(inds,inds)],
            'bigL': sl.cho_factor(bigC[np.ix_(inds,inds)], lower=True),
            'mu': mu[inds],
            'rho': rho[inds,:],
            'gamma': gamma[inds],
            'inds': inds,
            'pairs': counts[ii],
        }

    return bin_data

def calculate_h_sigma_hat(mu, bigL, rho, narrow_bin=False, **kwargs):
    """If narrow_bin = True, don't include one term of mu_ef"""
    if narrow_bin:
        xi = sl.cho_solve(bigL, np.ones_like(mu))
        den = np.sum(xi*np.ones_like(mu))

    else:
        xi = sl.cho_solve(bigL, mu)
        den = np.sum(xi*mu)

    num = np.sum(xi[:,None]*rho, axis=0)
    return np.mean(num/den), (1/den)/(rho.shape[1])

def get_optimal_h(corrmat, log10_Arn, dt, method='all', log10_h_min=-2, log10_h_max=3, nbins=200):
    """Can be improved: perhaps first optimize, and then grid?

    For Noise-Marginalized OS, we just use MCMC chaink so this function is not used
    """
    x_log10_h = np.linspace(log10_h_min, log10_h_max, nbins)
    rho = correlate_data(dt, log10_Arn, method=method)

    h2_trials = 10**x_log10_h
    h2_hats = []
    sigma_h2 = []

    for log10_h in x_log10_h:
        bigC, mu = build_bigC_mu(corrmat, log10_hc=log10_h, log10_Arn=log10_Arn, method=method)

        h2_hat, s2h2_hat = calculate_h_sigma_hat(mu, sl.cho_factor(bigC, lower=True), rho)
        h2_hats.append(h2_hat)
        sigma_h2.append(np.sqrt(s2h2_hat))

    h2_hats = np.array(h2_hats)
    sigma_h2 = np.array(sigma_h2)

    # Find where consistent
    msk_cons = np.where(np.logical_and(
        h2_trials<h2_hats+sigma_h2,
        h2_trials>h2_hats-sigma_h2
    ))

    if(msk_cons[0]==True):
        return 0, sigma_h2[0]

    elif np.sum(msk_cons)==0:
        print(h2_trials, h2_hats, sigma_h2)
        raise ValueError("No appropriate values found!")

    else:
        ind = np.argmin(np.abs(h2_trials-h2_hats)[msk_cons])

        return h2_hats[msk_cons][ind], sigma_h2[msk_cons][ind]

def get_narrowbin_estimator(bigC, mu, rho, psrpos):
    """Estimator for cross-power per separation bin"""
    method = 'cross'

    mu, rho, bigC, gamma = order_by_separation(mu, rho, bigC, psrpos, method=method)
    bin_data = subselect_by_bin(mu, rho, bigC, gamma, nbins=15)

    for i_bin, bindict in bin_data.items():
        h2hat, sigma2_h2hat = calculate_h_sigma_hat(**bindict, narrow_bin=True)

        bindict['crosspower'] = h2hat
        bindict['crosspower_err'] = np.sqrt(sigma2_h2hat)

    psr_sep = np.array([bd['bin'] for bd in bin_data.values()])
    cross_power = np.array([bd['crosspower'] for bd in bin_data.values()])
    cross_power_err = np.array([bd['crosspower_err'] for bd in bin_data.values()])

    return psr_sep, cross_power, cross_power_err

def get_ds(log10_hc, log10_Arn, corrmat, dt, snr=True):
    """Anholm et al detection statistic"""
    rho_ab = correlate_data(dt, log10_Arn, method='cross')
    sigma = np.sqrt(10**log10_Arn + 10**log10_hc)
    a, b = get_indices(len(corrmat), method='cross')
    Gamma_ab = corrmat[a,b][:,None]
    sigma2_ab = (sigma[a]*sigma[b])[:,None]

    num = np.sum(Gamma_ab * rho_ab / sigma2_ab, axis=0)
    den = np.sum(Gamma_ab**2 / sigma2_ab)

    h2hat = num / den
    sigma2 = 1 / den

    if snr:
        return np.mean(h2hat) / np.sqrt(sigma2/len(num))
    else:
        return np.mean(h2hat), sigma2/len(num)

def get_os(log10_hc, log10_Arn, corrmat, dt, psrpos, method='cross'):
    """PCOS estimator for h2"""
    bigC, mu = build_bigC_mu(corrmat, log10_hc=log10_hc, log10_Arn=log10_Arn, method=method)
    rho = correlate_data(dt, log10_Arn, method=method)

    if method=='cross':
        psr_sep, cross_power, cross_power_err = get_narrowbin_estimator(bigC, mu, rho, psrpos)
    else:
        psr_sep, cross_power, cross_power_err = None, None, None

    h2hat, sigma2_h2hat = calculate_h_sigma_hat(mu, sl.cho_factor(bigC, lower=True), rho)

    return psr_sep, cross_power, cross_power_err, h2hat, sigma2_h2hat

def get_nmos(log10_hc_samples, log10_Arn_samples, corrmat, dt, psrpos, method='cross'):
    """Noise-marginaled PCOS"""
    psr_sep_l, cross_power_l, cross_power_err_l = [], [], []
    h2hat_l, sigma2_h2hat_l = [], []

    for log10_hc_sample, log10_Arn_sample in zip(log10_hc_samples, log10_Arn_samples):
        bigC, mu = build_bigC_mu(corrmat, log10_hc=log10_hc_sample, log10_Arn=log10_Arn_sample, method=method)
        rho = correlate_data(dt, log10_Arn_sample, method=method)

        if method=='cross':
            psr_sep, cross_power, cross_power_err = get_narrowbin_estimator(bigC, mu, rho, psrpos)
        else:
            psr_sep, cross_power, cross_power_err = None, None, None

        h2hat, sigma2_h2hat = calculate_h_sigma_hat(mu, sl.cho_factor(bigC, lower=True), rho)

        psr_sep_l.append(psr_sep)
        cross_power_l.append(cross_power)
        cross_power_err_l.append(cross_power_err)

        h2hat_l.append(h2hat)
        sigma2_h2hat_l.append(sigma2_h2hat)

    if method=='cross':
        psr_sep_nmos = psr_sep_l[0]
        cross_power_nmos = np.mean(cross_power_l, axis=0)
        cross_power_err_nmos = np.sqrt(np.sum(np.vstack(cross_power_err_l)**2, axis=0)/len(cross_power_err_l))

    else:
        psr_sep_nmos, cross_power_nmos, cross_power_err_nmos = None, None, None

    h2hat_nmos = np.mean(h2hat_l)
    sigma2_h2hat_nmos = np.mean(sigma2_h2hat_l)

    return psr_sep_nmos, cross_power_nmos, cross_power_err_nmos, h2hat_nmos, sigma2_h2hat_nmos

def get_nmds(log10_hc_samples, log10_Arn_samples, corrmat, dt):
    """Noise-marginaled PCOS"""
    h2s, s2s = [], []

    for log10_hc_sample, log10_Arn_sample in zip(log10_hc_samples, log10_Arn_samples):
        h2hat, sigma2 = get_ds(log10_hc_sample, log10_Arn_sample, corrmat, dt, snr=False)
        h2s.append(h2hat)
        s2s.append(sigma2)

    h2hat = np.mean(h2s)
    sigma2 = np.mean(sigma2)

    return h2hat / np.sqrt(sigma2)

def set_mpl_rcparams():
    plot_rcparams = {
        "backend": "module://matplotlib_inline.backend_inline",
        #"backend": "pdf",
        "axes.labelsize": 20,
        "lines.markersize": 4,
        "font.size": 16,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.markeredgewidth": 1,
        "axes.linewidth": 1.2,
        "legend.fontsize": 16,
        "xtick.labelsize": 12,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.major.top": True,
        "xtick.minor.top": True,
        "ytick.labelsize": 12,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.major.right": True,
        "ytick.minor.right": True,
        "savefig.dpi": 400,
        "path.simplify": True,
        "font.family": "serif",
        "font.serif": "Times",
        "text.usetex": True,
        "figure.figsize": [10.0, 7.0]}

    mpl.rcParams.update(plot_rcparams)

def loglik(x, amp, log10_Awn):
    logamp = np.log10(10**amp + 10**log10_Awn)
    return np.sum(sstats.norm(loc=0, scale=10**(0.5*logamp)).logpdf(x))

def rn_bayes_factor(r, amin, amax, log10_Awn, nsamples=10000):
    x = np.linspace(amin, amax, nsamples)
    yl = np.stack([loglik(r, xx, log10_Awn) for xx in x])
    y = np.exp(yl - np.max(yl))

    return np.mean(y) / y[0]

def get_1D_rn_plot(r, amin, amax, log10_Awn, nsamples=100):
    x = np.linspace(amin, amax, nsamples)
    yl = np.array([loglik(r, xx, log10_Awn) for xx in x])

    y = np.exp(yl - np.max(yl))

    return x, y

def logliks(x, amp, log10_Awn):
    logamp = np.log10(10**amp + 10**log10_Awn)
    return np.sum(sstats.norm(loc=0, scale=10**(0.5*logamp)).logpdf(x), axis=1)

def rn_bayes_factors(r, amin, amax, log10_Awn, nsamples=10000):
    x = np.linspace(amin, amax, nsamples)
    yl = np.vstack([logliks(r, xx, log10_Awn) for xx in x])

    y = np.exp(yl - np.max(yl, axis=0)[None,:])

    return np.mean(y, axis=0) / y[0,:]

def get_rn_estimates(r, amin, amax, log10_Awn, nsamples=10000, bfth=3):
    x = np.linspace(amin, amax, nsamples)
    yl = np.vstack([logliks(r, xx, log10_Awn) for xx in x])

    maxinds = np.argmax(yl, axis=0)
    y = np.exp(yl - np.max(yl, axis=0)[None,:])

    bfs = np.mean(y, axis=0) / y[0,:]

    xmax = x[maxinds]
    xmax[bfs<bfth] = amin

    return xmax, bfs

def log_likelihood(log10_Arn, log10_hc, log10_Awn, corrmat, dt):
    C = np.diag(10**log10_Arn + 10**log10_Awn) + corrmat * 10**log10_hc
    cf = sl.cho_factor(C, lower=True)

    logdet = np.sum(np.log(np.diag(cf[0])))
    xi2 = np.sum(dt * sl.cho_solve(cf, dt), axis=0)

    return np.sum(-0.5*xi2 - logdet - np.log(2*np.pi))

def log_prior(log10_Arn, log10_hc, amin=-3, amax=5):
    if np.any(log10_Arn<amin) or log10_hc<amin:
        return -np.inf

    if np.any(log10_Arn>amax) or log10_hc>amax:
        return -np.inf

    return 0.0

def log_posterior(log10_Arn, log10_hc, log10_Awn, corrmat, dt):
    ll = log_likelihood(log10_Arn, log10_hc, log10_Awn, corrmat, dt)
    lp = log_prior(log10_Arn, log10_hc)
    return ll + lp


def run_model(m, N=5000):
    nuts_kernel = NUTS(m)
    mcmc = MCMC(nuts_kernel, num_samples=N, num_warmup=1000, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(0))
    return mcmc.get_samples()

if __name__ == '__main__':

    set_mpl_rcparams()

    seed = 1234567
    np.random.seed(seed)

    # In Fourier space, total = GW + white + red
    npsrs = 88
    nobs = 6
    log10_Awn = 1.5
    log10_Arn_prior_mu = 1.5
    log10_Arnt_prior_mu = np.log10(10**log10_Awn + 10**log10_Arn_prior_mu)
    log10_Arn_prior_sigma = 0.20001   # 0.00001
    log10_hc_prior_mu = 1.35

    phi, theta, dt, log10_Arn_true, log10_hc_true, log10_Arn_prior_mu, log10_Arn_prior_sigma = \
        generate_data_fixh(npsrs=npsrs, nobs=nobs, correlations='hd',
                        log10_Arn_prior_mu=log10_Arnt_prior_mu, log10_Arn_prior_sigma=log10_Arn_prior_sigma,
                        log10_hc_prior_mu=log10_hc_prior_mu, prior_dist=sstats.norm)

    psrpos = phitheta_to_psrpos(phi, theta)
    hdmat = hdcorrmat(psrpos, psrTerm=True)

    # Create RN estimates for 'Data' model
    log10_Arn_est, bfs = get_rn_estimates(dt[:,:], amin=-3, amax=5, log10_Awn=log10_Awn, bfth=3)
    bfbig = (bfs>3)
    bfs[:3], np.sum(bfbig)

    Nsamples = 1_000

    # Run 'Data' model
    np_model_data = get_model(dt, hdmat, model='Data', log10_Awn=1.5, log10_Arn=None, prior_type="Log-Uniform", corr_model="hd", log10_Arn_prior_mu=None, log10_Arn_prior_sigma=None, log10_hc_prior_mu=None, log10_hc_prior_sigma=None, bf_select=bfbig)

    samples_data = run_model(np_model_data, N=Nsamples)

    # Run 'ER' model
    np_model_er = get_model(dt, hdmat, model='ER', log10_Awn=1.5, log10_Arn=None, prior_type="Log-Uniform", corr_model="hd", log10_Arn_prior_mu=None, log10_Arn_prior_sigma=None, log10_hc_prior_mu=None, log10_hc_prior_sigma=None, bf_select=None)

    samples_er = run_model(np_model_er, N=Nsamples)

    # Run 'HBM' model
    np_model_hbm = get_model(dt, hdmat, model='ER', log10_Awn=1.5, log10_Arn=None, prior_type="Hierarchical", corr_model="hd", log10_Arn_prior_mu=None, log10_Arn_prior_sigma=None, log10_hc_prior_mu=None, log10_hc_prior_sigma=None, bf_select=None)

    samples_hbm = run_model(np_model_hbm, N=Nsamples)

    # Make figure 2:
    fig, ax = plt.subplots(figsize=(8,5))

    sns.kdeplot(samples_hbm['log10_hc'], label='HBM (M3)');
    sns.kdeplot(samples_er['log10_hc'], label='Full (M1)');
    sns.kdeplot(samples_data['log10_hc'], label='Data (M2)');

    ax.vlines(log10_hc_true, ymin=0, ymax=6.0, colors='k', ls='--', label='Injection')

    ax.set_xlabel(r"$\log_{10}h$")
    ax.set_ylabel("Density")
    ax.set_title("GWB amplitude by model")

    ax.set_xlim([0, 2])
    ax.set_ylim([0, 6])

    ax.legend()
    fig.savefig(f'fig2-hcpost.pdf', dpi=300)


    # Optimize to get MAP
    x0 = np.hstack([log10_Arn_true, [log10_hc_true]])
    get_ll = lambda x: -log_posterior(x[:-1], x[-1], log10_Awn, hdmat, dt)
    res = sopt.minimize(get_ll, x0, method='Nelder-Mead', options={'maxiter': 500000})

    # CrossPower takes RN+WN. Convert the values
    log10_Arn_est = np.log10(10**res.x[:-1] + 10**log10_Awn)
    log10_hc_est = res.x[-1]

    # Where no RN is there from BF, just set it to WN
    log10_Abf_est = np.copy(log10_Arn_est)
    log10_Abf_est[~bfbig] = log10_Awn

    # Calculate the NMOS
    nmos_thin = 100

    psr_sep_nmos, cross_power_nmos, cross_power_err_nmos, h2hat_nmos, sigma2_h2hat_nmos = get_nmos(samples_er['log10_hc'][::nmos_thin], samples_er['log10_Arn'][::nmos_thin], hdmat, dt, psrpos, method='cross')
    psr_sep_nmoshbm, cross_power_nmoshbm, cross_power_err_nmoshbm, h2hat_nmoshbm, sigma2_h2hat_nmoshbm = get_nmos(samples_hbm['log10_hc'][::nmos_thin], samples_hbm['log10_Arn'][::nmos_thin], hdmat, dt, psrpos, method='cross')


    # Make figure 1
    fig, ax = plt.subplots(figsize=(8,5))

    psr_sep, cross_power, cross_power_err, h2hat, sigma2_h2hat = get_os(log10_hc_true, log10_Arn_true, hdmat, dt, psrpos, method='cross')
    ax.errorbar((psr_sep-0.02)*180/np.pi, cross_power, yerr=cross_power_err, fmt='.', label="Injected")

    ax.errorbar((psr_sep_nmos+0.0)*180/np.pi, cross_power_nmos, yerr=cross_power_err_nmos, color='r', fmt='.', label="NMOS")

    psr_sep, cross_power, cross_power_err, h2hat, sigma2_h2hat = get_os(log10_hc_est, log10_Arn_est, hdmat, dt, psrpos, method='cross')
    ax.errorbar((psr_sep+0.02)*180/np.pi, cross_power, yerr=cross_power_err, fmt='.', label="MAP full")

    psr_sep, cross_power, cross_power_err, h2hat, sigma2_h2hat = get_os(log10_hc_est, log10_Abf_est, hdmat, dt, psrpos, method='cross')
    ax.errorbar((psr_sep+0.04)*180/np.pi, cross_power, yerr=cross_power_err, fmt='.', label="MAP data model")

    # Also plot H&D
    gamma = np.linspace(0, np.pi, 1000)
    ghd = hdfunc(gamma)
    ax.plot((gamma)*180/np.pi, ghd*10**log10_hc_true, 'k--', label='H\&D')

    ax.set_xlabel("Angular separation")
    ax.set_ylabel("Correlated Power")

    ax.legend()

    fig.savefig(f'fig1-crosspower.pdf', dpi=300)
