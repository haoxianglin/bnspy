import numpy as np
import astropy.units as u
from astropy.constants import c, M_sun
from scipy.integrate import quad
from .utils import qquad_sphavg

class BaseJet:
    def __init__(self, Eiso_c=1e52 * u.erg, LF_c=400, theta_j=np.pi/2, s_ej=np.inf):
        self.Eiso_c = Eiso_c
        self.LF_c = LF_c
        self.theta_j = theta_j
        self.s_ej = s_ej

    def profile(self, theta):
        '''jet profile (per polar)'''
        return 1

    def Eiso(self, theta):
        '''Isotropic-equivalent energy'''
        return self.Eiso_c * self.profile(theta)

    def LF(self, theta):
        '''Lorentz factor'''
        return 1 + (self.LF_c - 1) * self.profile(theta)

    def Miso(self, theta):
        if self.Eiso(theta).value == 0 or self.LF(theta) - 1. == 0.:
            return 0 * M_sun
        else:
            return (self.Eiso(theta)/c**2).to(M_sun)/(self.LF(theta) - 1.)

    def u4(self, theta):
        return np.sqrt(self.LF(theta)**2 - 1)

    @property
    def energy(self):
        '''total bipolar jet energy'''
        return 2 * qquad_sphavg(lambda theta, phi: self.Eiso(theta), theta_r=np.pi/2)[0]

    @property
    def mass(self):
        '''total bipolar jet mass'''
        return 2 * qquad_sphavg(lambda theta, phi: self.Miso(theta), theta_r=np.pi/2)[0]
    
    @property
    def fb(self):
        '''beaming fraction by theta_j'''
        return 1 - np.cos(self.theta_j)

    @property
    def fb_energy(self):
        '''beaming fraction by energy ratio'''
        return (self.energy / self.Eiso_c).cgs.value


class Gaussian(BaseJet):
    def __init__(self, Eiso_c=1e52 * u.erg, LF_c=400, theta_j=0.1, s_ej=np.inf):
        super().__init__(Eiso_c, LF_c, theta_j, s_ej)
        
    def profile(self, theta):
        return np.exp(-theta**2 / self.theta_j**2)


class DoubleGaussian(Gaussian):
    '''https://iopscience.iop.org/article/10.3847/1538-4357/abb404'''
    def __init__(self, Eiso_c=1e52 * u.erg, LF_c=400, theta_j=0.1, s_ej=np.inf, ratio=1e-5, theta_j2=1):
        super().__init__(Eiso_c, LF_c, theta_j, s_ej)
        self.ratio = ratio
        self.theta_j2 = theta_j2

    def profile(self, theta):
        return np.exp(-theta**2 / self.theta_j**2) + self.ratio * np.exp(-theta**2 / self.theta_j2**2)


class TopHat(BaseJet):
    def __init__(self, Eiso_c=1e52 * u.erg, LF_c=400, theta_j=0.1, s_ej=np.inf, k=np.inf):
        super().__init__(Eiso_c, LF_c, theta_j, s_ej)
        self.k = k

    @staticmethod
    def logistic(x, x0=0, k=np.inf):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def profile(self, theta):
        return self.logistic(self.theta_j - theta, k=self.k) + self.logistic(
            self.theta_j + theta, k=self.k) - 1


class PowerLaw(BaseJet):
    def __init__(self, Eiso_c=1e52 * u.erg, LF_c=400, theta_j=0.1, s_ej=np.inf, k=2):
        super().__init__(Eiso_c, LF_c, theta_j, s_ej)
        self.k = k

    def profile(self, theta):
        return (1 + (theta / self.theta_j)**self.k)**(-self.k)