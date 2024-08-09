import numpy as np
import astropy.units as u
from astropy.constants import c, m_e, m_p, m_n, e, h, sigma_T, M_sun
u_flx = u.erg / u.s / u.cm**2
mec2 = (m_e * c**2).to(u.MeV)
import bns
import emcee
import multiprocess as mp
from scipy.optimize import basinhopping, minimize, minimize_scalar

import emcee
import multiprocess as mp
import dill
dill.settings['recurse'] = True

mp.freeze_support()

class lightcurve():
    def __init__():
        self.table = Table(names=('t_min', 't_max', 'e_min', 'e_max', 'flux', 'flux_err'),
            units=(u.s, u.s, u.eV, u.eV, u.erg/u.s/u.cm**2, u.erg/u.s/u.cm**2))
        
    def fitting():
    
    def mcmc():
        
    def plot():


def log_likelihood(x):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
            
    kwargs = {
        'E_theta': lambda theta: 10**x[0] * u.erg * np.exp(-theta**2 / x[7]**2),
        'n0': 10**x[1] * u.cm**-3,
        'g_theta': lambda theta: 1 + (10**x[2]-1) * np.exp(-theta**2 / x[7]**2),
        'epsb': 10**x[3],
        'epse': 10**x[4],
        'pe': x[5],
        'fe': 10**x[6],
        's_amb': 0,
        'ssc': True,
        'ssa': True,
        'syn': True,
        'ebl': True,
        'eic': False,
        'gg': True,
        'electron_y': True,
    }

    @np.vectorize
    def model(e_obs, t_obs):
        return afterglow.gaussian_smart(
            e_obs*u.eV,
            t_obs*u.s,
            theta_obs=0.,
            z=0.1505,
            intbnd=intbnd_interp(t_obs*u.s, 0., 10 **
                                 x[2]).reshape(4)[[0, 2, 3]],
            **kwargs,).value

    logll = 0

    '''model_HAWC = model(1e12, t_HAWC)
    logll += -sum(model_HAWC**2 / (f_HAWC/2)**2)'''
    model_TeV = model(1e12, t_TeV)
    logll += -sum((model_TeV - f_TeV)**2 / (model_TeV * f_TeV)) / len(f_TeV)

    model_GeV = model(1e9, t_GeV)
    logll += -sum( (model_GeV - f_GeV)**2 / (model_GeV * f_GeV)) / len(f_GeV)
    
    model_keV = model(0.5e3, t_keV)
    logll += -sum( (model_keV - f_keV)**2 / (model_keV * f_keV)) / len(f_keV)

    # print('x =', [float('{0:.4g}'.format(xi)) for xi in x], 'Xsq =', '{0:.4g}'.format(-logll))
    return logll

def log_prior(x):
    bnds = ((50, 60), (-4, 4), (1, 4), (-6, 0.), (-6, 0.), (2, 3), (-4, 2.), (0, 0.4))
    if all([bnds[i][0] < x[i] < bnds[i][-1] for i in range(len(x))]):
        return 0.0
    return -np.inf

def log_probability(x):
    lp = log_prior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(x)

def mcmc():
    x0 = [55.55, -0.9632, 3.024, -2.407, -1.632, 2.396, 1.015, 0.04762]
    initial = np.load('chain.npy')[-1] # x0 + 1e-4 * np.random.randn(32, len(x0))
    nwalkers, ndim = initial.shape
    nsteps = 500

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True, store=True)
        end = time.time()
    multi_time = end - start