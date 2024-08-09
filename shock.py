# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import quad, dblquad, odeint
from scipy.special import hyp2f1, gamma
import astropy.units as u
from astropy.constants import c, m_e, m_p, e, h, sigma_T
from astropy.cosmology import Planck15

from functools import lru_cache, cached_property, wraps, WRAPPER_ASSIGNMENTS
from typing import Callable
import warnings

from .models import (
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton,
    EblAbsorptionModel,
)

e = e.gauss
pcm3 = u.cm**(-3)
mec = m_e * c
mec2 = m_e * c**2
mpc2 = m_p * c**2
D_L = Planck15.luminosity_distance

cgs_B = u.Unit('cm(-1/2) g(1/2) s(-1)')
equiv_B = [(u.G, cgs_B, lambda x: x, lambda x: x)]

from .jet import BaseJet, Gaussian
from .utils import unit_extractor, cos_sph, trapz_sphavg, qquad_sphavg, trapz_loglog, extrap_loglog

def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def ignore_unhashable(func): 
    uncached = func.__wrapped__
    attributes = WRAPPER_ASSIGNMENTS + ('cache_info', 'cache_clear')
    wraps(func, assigned=attributes) 
    def wrapper(*args, **kwargs): 
        try: 
            return func(*args, **kwargs) 
        except TypeError as error: 
            if 'unhashable type' in str(error): 
                return uncached(*args, **kwargs) 
            raise 
    wrapper.__uncached__ = uncached
    return wrapper

import os
import sys
import contextlib

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

# to_do : 
# Cache decorator for numpy arrays
# _u4 to staticmethod
# _isoeq simplify
# np.vstack
# 0*u.G
# amp = np.inf
# magnetic field unit
# cached_property for ShockRadiation ? 

class Shock:
    def __init__(
        self,
        jet: BaseJet = Gaussian(Eiso_c=1e54 * u.erg, LF_c=1000, theta_j=0.1, s_ej=np.inf),
        n0: u.Quantity = 1 * pcm3,
        s_amb: float = 0.,
        epsb: float = 0.0001,
        epse: float = 0.1,
        pe: float = 2.5,
        fe: float = 0.1,
        injection = None,
    ):
        self.jet = jet
        self.n0 = n0
        self.s_amb = s_amb

        self.epsb = epsb
        self.epse = epse
        self.pe = pe
        self.fe = fe

        self.injection = injection

        # https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
        self.Rd_map = lru_cache()(self._Rd_map)
        self.u4_map = lru_cache()(self._u4_map)
        self.u4_map_grb221009A = lru_cache()(self._u4_map_grb221009A)
        self.isoeq = lru_cache()(self._isoeq)
        self.eats = lru_cache()(self._eats)

    @property
    def _Rd_map(self):
        def _map(theta: float) -> u.Quantity:
            Eiso = self.jet.Eiso(theta)
            LF = self.jet.LF(theta)
            if LF-1. < 1e-6 or Eiso.value == 0:
                return np.inf * u.pc
            else:
                return (((3 - self.s_amb) * Eiso /
                        (4. * np.pi * self.n0 * m_p * c**2 *
                        (LF**2 - 1)))**(1 / (3 - self.s_amb))).to(u.pc)
        return _map

    @property
    def _u4_map(self):
        def _map(theta: float) -> Callable:
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            g0 = self.jet.LF(theta)
            u0, b0 = np.sqrt(g0**2 - 1), np.sqrt(1 - 1 / g0**2)

            if np.isposinf(self.jet.s_ej):
                def u4_x(x):
                    if x < 1e-2: # affects eats shape but not flux
                        u4 = u0
                    elif x > 1e2:
                        u4 = u0 * x**((self.s_amb - 3) / 2)
                    else:
                        x3 = 2 / (g0 + 1) * x**(3 - self.s_amb)
                        g = max(1, (np.sqrt(1 + 2 * g0 * x3 + x3**2) - 1) / x3)
                        u4 = np.sqrt(g**2 - 1)
                    return u4
            else:
                def u4_x(x):
                    
                    def f_Eiso_stratified(u4, u_min=u0, u_max=np.inf, s=self.jet.s_ej):
                        u4 = max(u_min, min(u_max, u4))
                        if np.isposinf(s):
                            # return 0. if u4 > u_min else 1.
                            # return (u4 / u_min)**-np.inf if u4 > u_min
                            # f(u_min) = 1, f(u_min+) = 0, thus
                            # (int -df/du du from u_min to u_min+) returns 1
                            return np.heaviside(u_min-u4, 1.)
                        return (u4**(-s) - u_max**(-s)) / (u_min**(-s) - u_max**(-s))
                    
                    def f_Miso_stratified(u4, u_min=u0, u_max=np.inf, s=self.jet.s_ej):
                        u4 = max(u_min, min(u_max, u4))
                        print(u4, u_min, u_max, s)
                        if np.isposinf(s):
                            return 1 / (np.sqrt(u_min**2+1)-1) if u <= u_min else 0.
                        return quad(lambda _u: s * _u**(-s-1) / (np.sqrt(_u**2+1)-1), u4, u_max)[0] / (u_min**(-s) - u_max**(-s))
                        '''def m(u, s):
                            if not s % 2:
                                s += 1e-6
                            return s / (s + 2) * u**(-s-2) * (1 + hyp2f1(-1/2, -1-s/2, -s/2, -u**2))
                        return (m(u, s) - m(u_max, s)) / (u_min**(-s) - u_max**(-s))'''
                        '''uu = np.logspace(np.log10(u), np.log10(u_max), 10)
                        return trapz_loglog(s * uu**(-s-1) / (np.sqrt(uu**2+1)-1), uu) / (u_min**(-s) - u_max**(-s))'''

                    e_amb = lambda x, _u: x**(3 - self.s_amb) * (_u / u0)**2
                    u4 = fsolve(lambda _u: - f_Eiso_stratified(_u[0]) + e_amb(x, _u[0]), u0)[0] # + f_Miso_stratified(_u[0]) * (np.sqrt(_u[0]**2 + 1) - 1)
                    return u4
            return np.vectorize(u4_x, otypes=[float])    
        return _map

    @staticmethod
    def _eats(
        x_obs: float,  # c * t_obs / (1 + z) / self.Rd_map(theta)
        theta_D: float, 
        dyn: Callable # self.u4_map(theta)
    ) -> float:

        def func(x):
            return - x_obs - np.cos(theta_D) * x + quad(
                lambda _x: np.sqrt(dyn(_x)**2 + 1) / dyn(_x), 0, x)[0]

        x = fsolve(func, x_obs)[0]

        return x
    
    def trgd_map(self, t_obs=0*u.s, z=0):
        def _map(theta, theta_D):

            Rd = self.Rd_map(theta)
            u4_x = self.u4_map(theta)
            x = self.eats((c * t_obs / (1 + z) / Rd).cgs.value,
                theta_D,
                u4_x
                )
            R = x * self.Rd_map(theta)
            
            u4 = u4_x(x)
            LF = np.sqrt(1 + u4**2)
            b = u4 / np.sqrt(1 + u4**2)
            Doppler = (LF * (1 - b * np.cos(theta_D)))**-1

            # t = (Rd / c).cgs * quad(lambda _x: np.sqrt(1+u4_x(_x)**2) / u4_x(_x), 0, x)[0]
            # tsh = (Rd / c).cgs * quad(lambda _x: 1 / u4_x(_x), 0, x)[0]
            tsh = (Rd / c).cgs * x / LF

            return [tsh, R, LF, Doppler]

        return _map

    @property
    def _u4_map_grb221009A(self):
        def _map(theta: float, theta_D: float) -> Callable:
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            g0 = self.jet.LF(theta)

            T_hebs, F_hebs = np.loadtxt("../hebsGRD1_lowgain_1s.txt", unpack=True)
            T_hebs = T_hebs - 225
            F_hebs =  F_hebs - min(F_hebs)
            band_hebs = [0.4, 6] * u.MeV
            # https://arxiv.org/pdf/2303.01203.pdf
            area_hebs = 45 * u.cm**2

            z = 0.1505
            d = Planck15.luminosity_distance(z)
            L_hebs = (F_hebs * 1/u.s * np.mean(band_hebs, axis=-1) / area_hebs * 4*np.pi* d**2).to(u.erg/u.s).value

            def L(T):
                return np.interp(T, T_hebs, L_hebs)

            def model(sol, T):
                E = sol[0]
                R = sol[1]

                b0 = np.sqrt(1-1/g0**2)
                u0 = b0*g0
                
                if E==0 or R==0:
                    b = b0
                else:
                    M = 4/(3-self.s_amb) * np.pi * R**(3-self.s_amb) * self.n0.cgs.value * m_p.cgs.value
                    u4 = u0 * np.sqrt( E/(E + 2*g0*(g0-1)*M*c.cgs.value**2) )
                    b = u4/np.sqrt(u4**2+1)

                DT = (1-b0*np.cos(theta_D))/b0 * R/c.cgs.value * (1+z)
                
                dEdT = L((T - DT)/(1+z)) * ( 1 - b/b0 * (1-b0*np.cos(theta_D))/(1-b*np.cos(theta_D)) * 0.999 ) / (1+z)
                dRdT = b * c.cgs.value / (1-b*np.cos(theta_D)) / (1+z)
                
                return [dEdT, dRdT]
            
            def ode_calc(x, model):
                y = []
                y.append([0, 0])

                for i in range(1, len(x)):
                    xspan = [x[i-1],x[i]]
                    with stdout_redirected():
                        sol = odeint(model, y[-1], xspan)
                    y.append(sol[1])
                
                return y
            
            _T = np.linspace(180-225, (600-225)*2, 100) # * u.s
            _y = ode_calc(_T, model=model)
            _E = np.array(_y)[:,0]
            _R = np.array(_y)[:,1]

            _dRdcT = np.diff(_R)/np.diff(_T)/c.cgs.value
            _b = _dRdcT/(1+_dRdcT)
            _g = 1/np.sqrt(1-_b**2)
            _u = _b*_g
            _D = 1/(1-_b)/_g
            
            _R = (_R[:-1]+_R[1:])/2
            _T = (_T[:-1]+_T[1:])/2
            _E = (_E[:-1]+_E[1:])/2
            _t = _T/(1+z) + _R*np.cos(theta_D)/c.cgs.value

            return lambda x: extrap_loglog(x, _R/self.Rd_map(theta).cgs.value, _u)

        return _map

    def trgd_map_grb221009A(self, t_obs=0*u.s, z=0):
        def _map(theta, theta_D):

            Rd = self.Rd_map(theta)
            u4_x = self.u4_map_grb221009A(theta, theta_D)

            x = self.eats((c * t_obs / (1 + z) / Rd).cgs.value,
                theta_D,
                u4_x
                )
            R = x * self.Rd_map(theta)
            
            u4 = u4_x(x)
            # t = (Rd / c).cgs * quad(lambda _x: np.sqrt(1+u4_x(_x)**2) / u4_x(_x), 0, x)[0]
            tsh = (Rd / c).cgs * quad(lambda _x: 1 / u4_x(_x), 0, x)[0]
            # u4_x = 0, tsh -> inf

            LF = np.sqrt(1 + u4**2)

            b = u4 / np.sqrt(1 + u4**2)
            Doppler = (LF * (1 - b * np.cos(theta_D)))**-1

            return [tsh, R, LF, Doppler]
        return _map
    
    def afterglow_lc(
        self,
        e_obs: u.Quantity,
        t_obs: u.Quantity = np.logspace(0, 6, 15) * u.s,
        theta_obs: float = 0,
        z: float = 0,
        **kwargs
    ): 
        import matplotlib.pyplot as plt
        lc = unit_extractor([self.afterglow(e_obs, time, theta_obs, z) for time in t_obs])
        plt.loglog(t_obs, lc)

    def afterglow_spec(
        self,
        t_obs: u.Quantity,
        e_obs: u.Quantity = np.logspace(0, 14, 15) * u.eV,
        theta_obs: float = 0,
        z: float = 0,
        **kwargs
    ): 
        import matplotlib.pyplot as plt
        spec = unit_extractor([self.afterglow(energy, t_obs, theta_obs, z) for energy in e_obs])
        plt.loglog(e_obs, spec)
        plt.ylim([1e-15, None])

    def afterglow(
        self,
        e_obs: u.Quantity,
        t_obs: u.Quantity,
        theta_obs: float,
        z: float,
        method: str = 'trapz',
        intbnd: list = [],
        theta_step: int = 8, 
        phi_step: int = 3,
        **kwargs
    ) -> u.Quantity:
        method = method.lower()
        
        dfdOmega = lambda theta, phi: self.isoeq(e_obs=e_obs,
                                    t_obs=t_obs,
                                    theta_D=np.arccos(cos_sph(theta, phi, theta_obs, 0)),
                                    z=z,
                                    theta=theta,
                                    **kwargs)

        if not intbnd:
            bnd_l, bnd_r, bnd_h = self.intbnd(t_obs, theta_obs, z)
        else:
            bnd_l, bnd_r, bnd_h = intbnd

        if method == 'quad':
            return qquad_sphavg(dfdOmega, theta_l=bnd_l, theta_r=bnd_r, phi_h=bnd_h*2)[0]
        elif method == 'trapz':
            return trapz_sphavg(dfdOmega, theta_l=bnd_l, theta_r=bnd_r, phi_h=bnd_h*2, theta_step=theta_step, phi_step=phi_step)
        else:
            raise ValueError('Unknown method %s' % method)

    def intbnd(self,
               t_obs,
               theta_obs,
               z,
               zoom_ratio=2,
               zoom_step=50,
               zoom_edge=0.5):
        warnings.filterwarnings('ignore')

        _Doppler = lambda theta, phi: self.trgd_map(t_obs, z)(theta, np.arccos(cos_sph(theta, phi, theta_obs, 0)))[-1]

        # bound center
        bnd_c_lim = theta_obs
        bnd_c_iter = 0
        bnd_c = minimize_scalar(lambda theta: -_Doppler(theta, 0),
                                method='Bounded',
                                bounds=(-bnd_c_lim, bnd_c_lim))
        while bnd_c.fun == -1.0:
            bnd_c_iter += 1
            bnd_c_lim /= zoom_ratio
            bnd_c = minimize_scalar(lambda theta: -_Doppler(theta, 0),
                                    method='Bounded',
                                    bounds=(-bnd_c_lim, bnd_c_lim))
            if bnd_c_iter > zoom_step:
                break
        bnd_c = abs(bnd_c.x)
        Doppler_max = _Doppler(bnd_c, 0)

        # bound left
        bnd_l = minimize_scalar(
            lambda theta: (zoom_edge - _Doppler(theta, 0) / Doppler_max)**2,
            method='Bounded',
            bounds=(0., bnd_c))
        bnd_l = bnd_l.x

        # bound right
        bnd_r_lim = np.pi / 2
        bnd_r_iter = 0
        bnd_r = minimize_scalar(
            lambda theta: (zoom_edge - _Doppler(theta, 0) / Doppler_max)**2,
            method='Bounded',
            bounds=(bnd_c, min(np.pi / 2, bnd_r_lim)))
        while bnd_r.fun > zoom_edge / 10:
            bnd_r_iter += 1
            bnd_r_lim = (bnd_r_lim + bnd_c) / zoom_ratio
            bnd_r = minimize_scalar(
                lambda theta:
                (zoom_edge - _Doppler(theta, 0) / Doppler_max)**2,
                method='Bounded',
                bounds=(bnd_c, min(np.pi / 2, bnd_r_lim)))
            if bnd_r_iter > zoom_step:
                break
        bnd_r = bnd_r.x

        # bound height
        bnd_h = minimize_scalar(
            lambda phi: (zoom_edge - _Doppler(bnd_c, phi) / Doppler_max)**2,
            method='Bounded',
            bounds=(-np.pi, np.pi))
        if bnd_h.fun > zoom_edge / 10:
            bnd_h = np.pi
        else:
            bnd_h = abs(bnd_h.x)

        return bnd_l, bnd_r, bnd_h

    # !!! do not make phi as arg of isoeq !!! Use theta_D for caching !!!
    def _isoeq(
        self,
        e_obs: u.Quantity,
        t_obs: u.Quantity,
        theta_D: float,
        z: float,
        theta: float,
        syn: bool = True,
        ssc: bool = True,
        ssa: bool = True,
        gg: bool = True,
        ebl: bool = True,
        eic: bool = False,
        E_eic_obs: u.Quantity = None,
        L_eic_obs: u.Quantity = None,
    ):
        e_dim = np.ndim(np.array(e_obs, dtype=object))
        if e_dim == 0:
            e_obs = np.array([e_obs.value]) * e_obs.unit
        sed = np.zeros(len(e_obs)) 
        sed *= u.erg/u.s if z == 0 else u.erg/u.s/u.cm**2
        
        Rd = self.Rd_map(theta)
        
        if np.isinf(Rd) or self.pe <= 2.0:
            if e_dim == 0:
                sed = sed[0]
            return sed

        if self.injection == 'GRB221009A':
            tsh, R, LF, Doppler = self.trgd_map_grb221009A(t_obs, z)(theta, theta_D)
        else:
            tsh, R, LF, Doppler = self.trgd_map(t_obs, z)(theta, theta_D)

        if LF-1. < 1e-6 or R.value == 0.0 or R.value == np.inf:
            if e_dim == 0:
                sed = sed[0]
            return sed
        
        shrad = ShockRadiation(
            LF = LF,
            R = R,
            t = tsh,
            n = self.n0 * R**-self.s_amb,
            epsb = self.epsb,
            epse = self.epse,
            pe = self.pe,
            fe = self.fe,
        )

        e_sh = (1 + z) * e_obs / Doppler
        distance = D_L(z)

        '''
        Base = 2.52 * u.ms
        Syn  = 12.9 * u.ms
        SSC  = 25.7 * u.ms
        SSA  = 14.2 * u.ms
        GG   = 34.8 * u.ms
        EBL  = 14.3 * u.ms
        '''
        
        if syn:
            sed += Doppler**4 * shrad.SYN.sed(e_sh,
                                        distance)
        if ssc:
            sed += Doppler**4 * shrad.SSC.sed(e_sh,
                                        distance)

        if eic and (E_eic_obs is not None) and (L_eic_obs is not None):
            sed += Doppler**4 * shrad.EIC(
                E_eic=E_eic_obs / Doppler,  # shock comoving frame
                phn_eic=(L_eic_obs /  # L [1/eV/s] is Lorentz invariant
                        (4 * np.pi * R**2 * c) * 2.24).to(
                            1 / u.cm**3 / u.eV)).sed(e_sh, distance)

        if ssa:
            sed *= shrad.SSA(e_sh)

        if gg:
            sed *= shrad.GG(e_sh)

        if ebl:
            sed *= EblAbsorptionModel(redshift=z).transmission(
                e=e_obs)[0]
        
        if e_dim == 0:
            sed = sed[0]
        return sed


class ShockRadiation(object):
    def __init__(
        self,
        LF=1000,
        R=1e17 * u.cm,
        t=1e4 * u.s,
        n=1 * pcm3,
        epsb=0.0001,
        epse=0.1,
        pe=2.5,
        fe=0.1,
    ):

        self.LF = LF
        self.R = R
        self.t = t
        self.n = n
        self.epsb = epsb
        self.epse = epse
        self.pe = pe
        self.fe = fe

    @property
    def B(self):
        specific_heat_ratio = 4 / 3 + 1 / self.LF
        compression_ratio = (specific_heat_ratio * self.LF + 1) / (specific_heat_ratio - 1)

        _B = np.sqrt(8 * np.pi * self.epsb * compression_ratio * self.n *
                        mpc2 * (self.LF - 1))

        return _B.to((u.erg * pcm3)**(1 / 2))
    
    @property
    def electrons(self):
        gm = (self.epse / self.fe * (self.pe - 2) / (self.pe - 1) * m_p /
                m_e * (self.LF - 1)).cgs

        _gc = (6 * np.pi * mec / (sigma_T * self.B**2 * self.t)).cgs
        Y = (-1 + np.sqrt(1 + 4 * min(1, (gm / _gc)**(self.pe - 2)) *
                            self.epse / self.epsb)) / 2
        gc = _gc / (1 + Y)

        g_syn_max = np.sqrt(6 * np.pi * e / (sigma_T * self.B * (1 + Y))).cgs
        g_live_max = (e * self.B * self.t / mec).cgs
        g_esc_max = ((e * self.B * self.R) / (12 * self.LF * mec2)).cgs

        g_syn_min = 2.

        g0 = min(gm, gc)
        g_break = max(gm, gc)
        p1 = self.pe if gm < gc else 2
        p2 = self.pe + 1
        g_cutoff = min(g_syn_max, g_live_max, g_esc_max) 
        beta_cutoff = 2 if g_syn_max < min(g_live_max, g_esc_max) else 1

        Ne = self.fe * self.n * 4 / 3 * np.pi * self.R**3
        # amp = (Ne/mec2/( g0/(p1-1) + g_break*(g_break/g0)**(-p1)*(p2-p1)/(p1-1)/(p2-1) )).to(1/u.eV)
        amp = (p1 - 1) * Ne / mec2 * min(1 / g0, g0**(p1-1) / g_syn_min**p1)

        _electrons = ExponentialCutoffBrokenPowerLaw(
            amplitude=amp.to(1 / u.eV),
            e_0=(max(g_syn_min, g0) * mec2).to(u.eV),
            e_break=(max(g_syn_min, g_break) * mec2).to(u.eV),
            alpha_1=p1,
            alpha_2=p2,
            e_cutoff=(max(g_syn_min, g_cutoff) * mec2).to(u.eV),
            beta=beta_cutoff,
        )

        return _electrons

    @property
    def SYN(self):
        electrons = self.electrons
        return Synchrotron(
            electrons,
            B=self.B.value * u.G, 
            # https://github.com/astropy/astropy/issues?q=magnetic+field
            Eemax=electrons.e_cutoff * 2,
            Eemin=min(electrons.e_0, electrons.e_cutoff),
            nEed=50,
        )

    @property
    def SSC(self):
        electrons = self.electrons

        E_syn = np.logspace(-7, 9, 100) * u.eV  # shock comoving frame
        L_syn = self.SYN.flux(E_syn, distance=0 * u.cm)  # Doppler-Lorentz invariant
        if np.amax(L_syn.value) == 0. and self.R.value == 0.:
            phn_syn = np.zeros(len(L_syn)) / u.cm**3 / u.eV
        else:
            phn_syn = (L_syn / (4 * np.pi * self.R**2 * c) * 2.24).to(1 / u.cm**3 / u.eV)

        return InverseCompton(
            electrons,
            seed_photon_fields=[
                ["SSC", E_syn, phn_syn],
            ],
            Eemax=electrons.e_cutoff * 2,
            Eemin=min(electrons.e_0, electrons.e_cutoff),
            nEed=50,
        )

    def EIC(self, E_eic, phn_eic):
        electrons = self.electrons
        '''E_eic_obs = np.logspace(4, 9, 100) * u.eV
        E_eic = E_eic_obs / Doppler  # shock comoving frame
        # L_eic_obs / Doppler**2 / E_eic**2 = L_eic_obs / E_eic_obs**2  # Lorentz invariant
        L_eic_obs = (1.27e-5 * u.erg / u.cm**2 / u.s * 4 * np.pi * D_L(0.1505)**2 / E_eic_obs**2).to(1/u.s/u.eV)
        L_eic_obs *= (E_eic_obs.value / 1e8)**(-0.12) * np.exp(-E_eic_obs.value / 7.8e9) * np.exp(-t_obs.cgs.value / 100)
        phn_eic = (L_eic / (4 * np.pi * self.R**2 * c) * 2.24).to(1 / u.cm**3 / u.eV)'''
        return InverseCompton(
            electrons,
            seed_photon_fields=[
                ["EIC", E_eic, phn_eic],
            ],
            Eemax=electrons.e_cutoff * 2,
            Eemin=min(electrons.e_0, electrons.e_cutoff),
            nEed=50,
        )
    
    def SSA(self, e_sh):
        nu = e_sh / h
        p1 = self.electrons.alpha_1

        # eq.34 of Gould 1979, A&A, 76, 306
        Ke = self.electrons.amplitude * mec2
        nu_B = e * self.B / mec / (2 * np.pi)
        fp = np.pi**(1 / 2) * 3**((p1 + 1) / 2) / 8 * gamma(
            (3 * p1 + 22) / 12) * gamma((3 * p1 + 2) / 12) * gamma(
                (p1 + 6) / 4) / gamma((p1 + 8) / 4)
        kappa = Ke * (2 * np.pi * e / self.B).to(u.cm**2) * (
            (nu / nu_B).cgs)**(-(p1 + 4) / 2) * fp
        tau_ssa = (kappa / (4 * np.pi * self.R**2)).cgs

        # return tran_sph(tau_ssa)
        return np.exp(-tau_ssa)

    def GG(self, e_sh):
        # e_an_sh = e_an in shock comoving frame
        e_an_sh = (mec2**2 / e_sh).to(e_sh.unit)

        L_syn = self.SYN.flux(e_an_sh, distance=0 * u.cm)
        L_ssc = self.SSC.flux(e_an_sh, distance=0 * u.cm)
        if np.amax(L_syn.value) == 0. and np.amax(L_ssc.value) == 0. and self.R.value == 0.:
            n_gg = np.zeros(len(L_syn + L_ssc)) / u.cm**3
        else:
            n_gg = (e_an_sh * (L_syn + L_ssc) / 
                    (4 * np.pi * self.R**2 * c)).to(1 / u.cm**3)
        tau_gg = (sigma_T * n_gg * c * self.t).cgs
        '''
        phn_an = lambda e_: ((SYN.flux(unit_extractor(e_), distance=0 * u.cm) + SSC.
                                flux(unit_extractor(e_), distance=0 * u.cm)) /
                                (4 * np.pi * Rb**2 * c))[0].to(1 / u.cm**3 / u.eV)
        tau_gg = (3 / 8 * sigma_T * e_an_sh**2 * quad_loglog(
            lambda e_: e_**(-2) * phn_an(e_ * u.eV).to(1 / u.cm**3 / u.eV).value *
            phi_gg(e_ * u.eV / e_an_sh),
            e_an_sh.to(u.eV).value, 10000 * e_an_sh.to(u.eV).value)[0] * 1 /
                    u.cm**3 / u.eV**2 * c * tsh).cgs'''

        # return tran_sph(tau_gg)
        return np.exp(-tau_gg)
    
    @staticmethod
    def phi_gg(s0):
        b0 = np.sqrt(1 - 1 / s0)
        w0 = (1 + b0) / (1 - b0)
        L0 = quad(lambda w: np.log(w + 1) / w, 1, w0)[0] # -polylog(2, -w0)-np.pi**2/12
        return (1 + b0**2) / (1 - b0**2) * np.log(w0) - b0**2 * np.log(
            w0) - np.log(w0)**2 - 4 * b0 / (
                1 - b0**2) + 2 * b0 + 4 * np.log(w0) * np.log(w0 + 1) - L0

    @staticmethod
    def tran_sph(tau):
        # eq.38 of Gould 1979, A&A, 76, 306
        tau *= 2
        return np.where(
            tau < 1e-4, 1.0, 3 * (1 / 2 + np.exp(-tau) / tau -
                                (1 - np.exp(-tau)) / tau**2) / tau)