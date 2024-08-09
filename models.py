import logging
import os
import warnings
from collections import OrderedDict

import numpy as np
from astropy import units as u
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb
from astropy.utils.data import get_pkg_data_filename

from .utils import (
    memoize,
    validate_physical_type,
    validate_scalar,
    validate_array,
    trapz_loglog, 
    heaviside,
)

__all__ = [
    "ExponentialCutoffBrokenPowerLaw",
    "Synchrotron",
    "InverseCompton",
    "EblAbsorptionModel",
]

# add energy wrapper in _validate_ene
# 0 * u.G synchrotron output 0 flux

# Get a new logger to avoid changing the level of the astropy logger
# log = logging.getLogger("naima.radiative")
# log.setLevel(logging.INFO)

e = e.gauss

mec2 = (m_e * c ** 2).cgs
mec2_unit = u.Unit(mec2)

ar = (4 * sigma_sb / c).to("erg/(cm3 K4)")
r0 = (e ** 2 / mec2).to("cm")

u.def_physical_type(u.erg / u.cm ** 2 / u.s, "flux")
u.def_physical_type(u.Unit("1/(s cm2 erg)"), "differential flux")
u.def_physical_type(u.Unit("1/(s erg)"), "differential power")
u.def_physical_type(u.Unit("1/TeV"), "differential energy")
u.def_physical_type(u.Unit("1/cm3"), "number density")
u.def_physical_type(u.Unit("1/(eV cm3)"), "differential number density")

def _validate_ene(ene):
    from astropy.table import Table

    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'energy' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

        if np.ndim(ene) == 0:
            ene = np.array([ene.value]) * ene.unit

    return ene


class ExponentialCutoffBrokenPowerLaw:
    param_names = [
        "amplitude",
        "e_0",
        "e_break",
        "alpha_1",
        "alpha_2",
        "e_cutoff",
        "beta",
    ]
    _memoize = False
    _cache = {}
    _queue = []

    def __init__(
        self, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta=1.0
    ):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.e_break = validate_scalar(
            "e_break", e_break, domain="positive", physical_type="energy"
        )
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.e_cutoff = validate_scalar(
            "e_cutoff", e_cutoff, domain="positive", physical_type="energy"
        )
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        ee2 = e / e_cutoff
        return amplitude * K * (e / e_0) ** -alpha * np.exp(-(ee2 ** beta))


    @memoize
    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.e_break.to("eV").value,
            self.alpha_1,
            self.alpha_2,
            self.e_cutoff.to("eV").value,
            self.beta,
        )

    def __call__(self, e):
        """One dimensional broken power law model with exponential cutoff
        function"""
        e = _validate_ene(e)
        return self._calc(e)


class BaseRadiative:
    """Base class for radiative models

    This class implements the flux, sed methods and subclasses must implement
    the spectrum method which returns the intrinsic differential spectrum.
    """

    def __init__(self, particle_distribution):
        self.particle_distribution = particle_distribution
        try:
            # Check first for the amplitude attribute, which will be present if
            # the particle distribution is a function from naima.models
            pd = self.particle_distribution.amplitude
            validate_physical_type(
                "Particle distribution",
                pd,
                physical_type="differential energy",
            )
        except (AttributeError, TypeError):
            # otherwise check the output
            pd = self.particle_distribution([0.1, 1, 10] * u.TeV)
            validate_physical_type(
                "Particle distribution",
                pd,
                physical_type="differential energy",
            )

    @memoize
    def flux(self, photon_energy, distance=1 * u.kpc):
        """Differential flux at a given distance from the source.

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic differential
            luminosity will be returned. Default is 1 kpc.
        """

        spec = self._spectrum(photon_energy)

        if distance != 0:
            distance = validate_scalar(
                "distance", distance, physical_type="length"
            )
            spec /= 4 * np.pi * distance.to("cm") ** 2
            out_unit = "1/(s cm2 eV)"
        else:
            out_unit = "1/(s eV)"

        return spec.to(out_unit)

    def sed(self, photon_energy, distance=1 * u.kpc):
        """Spectral energy distribution at a given distance from the source.

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.
        """
        if distance != 0:
            out_unit = "erg/(cm2 s)"
        else:
            out_unit = "erg/s"

        photon_energy = _validate_ene(photon_energy)

        sed = (self.flux(photon_energy, distance) * photon_energy ** 2.0).to(
            out_unit
        )

        return sed


class BaseElectron(BaseRadiative):
    """Implements gam and nelec properties"""

    def __init__(self, particle_distribution):
        super().__init__(particle_distribution)
        self.param_names = ["Eemin", "Eemax", "nEed"]
        self._memoize = True
        self._cache = {}
        self._queue = []

    @property
    def _gam(self):
        """Lorentz factor array"""
        log10gmin = np.log10(self.Eemin / mec2).value
        log10gmax = np.log10(self.Eemax / mec2).value
        return np.logspace(
            log10gmin, log10gmax, int(self.nEed * (log10gmax - log10gmin))
        )

    @property
    def _nelec(self):
        """Particles per unit lorentz factor"""
        pd = self.particle_distribution(self._gam * mec2)
        return pd.to(1 / mec2_unit).value

    @property
    def We(self):
        """Total energy in electrons used for the radiative calculation"""
        We = trapz_loglog(self._gam * self._nelec, self._gam * mec2)
        return We

    def compute_We(self, Eemin=None, Eemax=None):
        """Total energy in electrons between energies Eemin and Eemax

        Parameters
        ----------
        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.
        """
        if Eemin is None and Eemax is None:
            We = self.We
        else:
            if Eemax is None:
                Eemax = self.Eemax
            if Eemin is None:
                Eemin = self.Eemin

            log10gmin = np.log10(Eemin / mec2).value
            log10gmax = np.log10(Eemax / mec2).value
            gam = np.logspace(
                log10gmin, log10gmax, int(self.nEed * (log10gmax - log10gmin))
            )
            nelec = (
                self.particle_distribution(gam * mec2).to(1 / mec2_unit).value
            )
            We = trapz_loglog(gam * nelec, gam * mec2)

        return We

    def set_We(self, We, Eemin=None, Eemax=None, amplitude_name=None):
        """Normalize particle distribution so that the total energy in electrons
        between Eemin and Eemax is We

        Parameters
        ----------
        We : :class:`~astropy.units.Quantity` float
            Desired energy in electrons.

        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.

        amplitude_name : str, optional
            Name of the amplitude parameter of the particle distribution. It
            must be accesible as an attribute of the distribution function.
            Defaults to ``amplitude``.
        """

        We = validate_scalar("We", We, physical_type="energy")
        oldWe = self.compute_We(Eemin=Eemin, Eemax=Eemax)

        if amplitude_name is None:
            try:
                self.particle_distribution.amplitude *= (
                    We / oldWe
                ).decompose()
            except AttributeError:
                ...
                # log.error(
                #     "The particle distribution does not have an attribute"
                #     " called amplitude to modify its normalization: you can"
                #     " set the name with the amplitude_name parameter of set_We"
                # )
        else:
            oldampl = getattr(self.particle_distribution, amplitude_name)
            setattr(
                self.particle_distribution,
                amplitude_name,
                oldampl * (We / oldWe).decompose(),
            )


class Synchrotron(BaseElectron):
    """Synchrotron emission from an electron population.

    This class uses the approximation of the synchrotron emissivity in a
    random magnetic field of Aharonian, Kelner, and Prosekin 2010, PhysRev D
    82, 3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).

    Parameters
    ----------
    particle_distribution : function
        Particle distribution function, taking electron energies as a
        `~astropy.units.Quantity` array or float, and returning the particle
        energy density in units of number of electrons per unit energy as a
        `~astropy.units.Quantity` array or float.

    B : :class:`~astropy.units.Quantity` float instance, optional
        Isotropic magnetic field strength. Default: equipartition
        with CMB (3.24e-6 G)

    Other parameters
    ----------------
    Eemin : :class:`~astropy.units.Quantity` float instance, optional
        Minimum electron energy for the electron distribution. Default is 1
        GeV.

    Eemax : :class:`~astropy.units.Quantity` float instance, optional
        Maximum electron energy for the electron distribution. Default is 510
        TeV.

    nEed : scalar
        Number of points per decade in energy for the electron energy and
        distribution arrays. Default is 100.
    """

    def __init__(self, particle_distribution, B=3.24e-6 * u.G, **kwargs):
        super().__init__(particle_distribution)
        self.B = validate_scalar("B", B, physical_type="magnetic flux density")
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = 100
        self.param_names += ["B"]
        self.__dict__.update(**kwargs)

    def _spectrum(self, photon_energy):
        """Compute intrinsic synchrotron differential spectrum for energies in
        ``photon_energy``

        Compute synchrotron for random magnetic field according to
        approximation of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82,
        3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` instance
            Photon energy array.
        """

        outspecene = _validate_ene(photon_energy)

        from scipy.special import cbrt

        def Gtilde(x):
            """
            AKP10 Eq. D7

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            Invoking crbt only once reduced time by ~40%
            """
            cb = cbrt(x)
            gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb ** 2.0)
            gt2 = 1 + 2.210 * cb ** 2.0 + 0.347 * cb ** 4.0
            gt3 = 1 + 1.353 * cb ** 2.0 + 0.217 * cb ** 4.0
            return gt1 * (gt2 / gt3) * np.exp(-x)

        # log.debug("calc_sy: Starting synchrotron computation with AKB2010...")

        # strip units, ensuring correct conversion
        # astropy units do not convert correctly for gyroradius calculation
        # when using cgs (SI is fine, see
        # https://github.com/astropy/astropy/issues/1687)

        if self.B.to("G").value == 0.:
            spec = np.zeros(len(outspecene)) / u.s / u.eV
            return spec

        CS1_0 = np.sqrt(3) * e.value ** 3 * self.B.to("G").value
        CS1_1 = (
            2
            * np.pi
            * m_e.cgs.value
            * c.cgs.value ** 2
            * hbar.cgs.value
            * outspecene.to("erg").value
        )
        CS1 = CS1_0 / CS1_1

        # Critical energy, erg
        Ec = (
            3
            * e.value
            * hbar.cgs.value
            * self.B.to("G").value
            * self._gam ** 2
        )
        Ec /= 2 * (m_e * c).cgs.value

        EgEc = outspecene.to("erg").value / np.vstack(Ec)
        dNdE = CS1 * Gtilde(EgEc)
        # return units
        spec = (
            trapz_loglog(np.vstack(self._nelec) * dNdE, self._gam, axis=0)
            / u.s
            / u.erg
        )
        spec = spec.to("1/(s eV)")

        return spec


def G12(x, a):
    """
    Eqs 20, 24, 25 of Khangulyan et al (2014)
    """
    alpha, a, beta, b = a
    pi26 = np.pi ** 2 / 6.0
    G = (pi26 + x) * np.exp(-x)
    tmp = 1 + b * x ** beta
    g = 1.0 / (a * x ** alpha / tmp + 1.0)
    return G * g


def G34(x, a):
    """
    Eqs 20, 24, 25 of Khangulyan et al (2014)
    """
    alpha, a, beta, b, c = a
    pi26 = np.pi ** 2 / 6.0
    tmp = (1 + c * x) / (1 + pi26 * c * x)
    G = pi26 * tmp * np.exp(-x)
    tmp = 1 + b * x ** beta
    g = 1.0 / (a * x ** alpha / tmp + 1.0)
    return G * g


class InverseCompton(BaseElectron):
    """Inverse Compton emission from an electron population.

    If you use this class in your research, please consult and cite
    `Khangulyan, D., Aharonian, F.A., & Kelner, S.R.  2014, Astrophysical
    Journal, 783, 100 <http://adsabs.harvard.edu/abs/2014ApJ...783..100K>`_

    Parameters
    ----------
    particle_distribution : function
        Particle distribution function, taking electron energies as a
        `~astropy.units.Quantity` array or float, and returning the particle
        energy density in units of number of electrons per unit energy as a
        `~astropy.units.Quantity` array or float.

    seed_photon_fields : string or iterable of strings (optional)
        A list of gray-body or non-thermal seed photon fields to use for IC
        calculation. Each of the items of the iterable can be either:

        * A string equal to ``CMB`` (default), ``NIR``, or ``FIR``, for which
          radiation fields with temperatures of 2.72 K, 30 K, and 3000 K, and
          energy densities of 0.261, 0.5, and 1 eV/cm³ will be used (these are
          the GALPROP values for a location at a distance of 6.5 kpc from the
          galactic center).

        * A list of length three (isotropic source) or four (anisotropic
          source) composed of:

            1. A name for the seed photon field.
            2. Its temperature (thermal source) or energy (monochromatic or
               non-thermal source) as a :class:`~astropy.units.Quantity`
               instance.
            3. Its photon field energy density as a
               :class:`~astropy.units.Quantity` instance.
            4. Optional: The angle between the seed photon direction and the
               scattered photon direction as a :class:`~astropy.units.Quantity`
               float instance.

    Other parameters
    ----------------
    Eemin : :class:`~astropy.units.Quantity` float instance, optional
        Minimum electron energy for the electron distribution. Default is 1
        GeV.

    Eemax : :class:`~astropy.units.Quantity` float instance, optional
        Maximum electron energy for the electron distribution. Default is 510
        TeV.

    nEed : scalar
        Number of points per decade in energy for the electron energy and
        distribution arrays. Default is 300.
    """

    def __init__(
        self, particle_distribution, seed_photon_fields=["CMB"], **kwargs
    ):
        super().__init__(particle_distribution)
        self.seed_photon_fields = self._process_input_seed(seed_photon_fields)
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = 100
        self.param_names += ["seed_photon_fields"]
        self.__dict__.update(**kwargs)

    @staticmethod
    def _process_input_seed(seed_photon_fields):
        """
        take input list of seed_photon_fields and fix them into usable format
        """

        Tcmb = 2.72548 * u.K  # 0.00057 K
        Tfir = 30 * u.K
        ufir = 0.5 * u.eV / u.cm ** 3
        Tnir = 3000 * u.K
        unir = 1.0 * u.eV / u.cm ** 3

        # Allow for seed_photon_fields definitions of the type 'CMB-NIR-FIR' or
        # 'CMB'
        if type(seed_photon_fields) != list:
            seed_photon_fields = seed_photon_fields.split("-")

        result = OrderedDict()

        for idx, inseed in enumerate(seed_photon_fields):
            seed = {}
            if isinstance(inseed, str):
                name = inseed
                seed["type"] = "thermal"
                if inseed == "CMB":
                    seed["T"] = Tcmb
                    seed["u"] = ar * Tcmb ** 4
                    seed["isotropic"] = True
                elif inseed == "FIR":
                    seed["T"] = Tfir
                    seed["u"] = ufir
                    seed["isotropic"] = True
                elif inseed == "NIR":
                    seed["T"] = Tnir
                    seed["u"] = unir
                    seed["isotropic"] = True
                else:
                    # log.warning(
                    #     "Will not use seed {0} because it is not "
                    #     "CMB, FIR or NIR".format(inseed)
                    # )
                    raise TypeError
            elif type(inseed) == list and (
                len(inseed) == 3 or len(inseed) == 4
            ):
                isotropic = len(inseed) == 3

                if isotropic:
                    name, T, uu = inseed
                    seed["isotropic"] = True
                else:
                    name, T, uu, theta = inseed
                    seed["isotropic"] = False
                    seed["theta"] = validate_scalar(
                        "{0}-theta".format(name), theta, physical_type="angle"
                    )

                thermal = T.unit.physical_type == "temperature"

                if thermal:
                    seed["type"] = "thermal"
                    validate_scalar(
                        "{0}-T".format(name),
                        T,
                        domain="positive",
                        physical_type="temperature",
                    )
                    seed["T"] = T
                    if uu == 0:
                        seed["u"] = ar * T ** 4
                    else:
                        # pressure has same physical type as energy density
                        validate_scalar(
                            "{0}-u".format(name),
                            uu,
                            domain="positive",
                            physical_type="pressure",
                        )
                        seed["u"] = uu
                else:
                    seed["type"] = "array"
                    # Ensure everything is in arrays
                    T = u.Quantity((T,)).flatten()
                    uu = u.Quantity((uu,)).flatten()

                    seed["energy"] = validate_array(
                        "{0}-energy".format(name),
                        T,
                        domain="positive",
                        physical_type="energy",
                    )

                    if np.isscalar(seed["energy"]) or seed["energy"].size == 1:
                        seed["photon_density"] = validate_scalar(
                            "{0}-density".format(name),
                            uu,
                            domain="positive",
                            physical_type="pressure",
                        )
                    else:
                        if uu.unit.physical_type == "pressure":
                            uu /= seed["energy"] ** 2
                        seed["photon_density"] = validate_array(
                            "{0}-density".format(name),
                            uu,
                            domain="positive",
                            physical_type="differential number density",
                        )
            else:
                raise TypeError(
                    "Unable to process seed photon"
                    " field: {0}".format(inseed)
                )

            result[name] = seed

        return result

    @staticmethod
    def _iso_ic_on_planck(
        electron_energy, soft_photon_temperature, gamma_energy
    ):
        """
        IC cross-section for isotropic interaction with a blackbody photon
        spectrum following Eq. 14 of Khangulyan, Aharonian, and Kelner 2014,
        ApJ 783, 100 (`arXiv:1310.7971 <http://www.arxiv.org/abs/1310.7971>`_).

        `electron_energy` and `gamma_energy` are in units of m_ec^2
        `soft_photon_temperature` is in units of K
        """
        Ktomec2 = 1.6863699549e-10
        soft_photon_temperature *= Ktomec2

        gamma_energy = np.vstack(gamma_energy)
        # Parameters from Eqs 26, 27
        a3 = [0.606, 0.443, 1.481, 0.540, 0.319]
        a4 = [0.461, 0.726, 1.457, 0.382, 6.620]
        z = gamma_energy / electron_energy
        x = z / (1 - z) / (4.0 * electron_energy * soft_photon_temperature)
        # Eq. 14
        cross_section = z ** 2 / (2 * (1 - z)) * G34(x, a3) + G34(x, a4)
        tmp = (soft_photon_temperature / electron_energy) ** 2
        # r0 = (e**2 / m_e / c**2).to('cm')
        # (2 * r0 ** 2 * m_e ** 3 * c ** 4 / (pi * hbar ** 3)).cgs
        tmp *= 2.6318735743809104e16
        cross_section = tmp * cross_section
        cc = (gamma_energy < electron_energy) * (electron_energy > 1)
        return np.where(cc, cross_section, np.zeros_like(cross_section))

    @staticmethod
    def _ani_ic_on_planck(
        electron_energy, soft_photon_temperature, gamma_energy, theta
    ):
        """
        IC cross-section for anisotropic interaction with a blackbody photon
        spectrum following Eq. 11 of Khangulyan, Aharonian, and Kelner 2014,
        ApJ 783, 100 (`arXiv:1310.7971 <http://www.arxiv.org/abs/1310.7971>`_).

        `electron_energy` and `gamma_energy` are in units of m_ec^2
        `soft_photon_temperature` is in units of K
        `theta` is in radians
        """
        Ktomec2 = 1.6863699549e-10
        soft_photon_temperature *= Ktomec2

        gamma_energy = gamma_energy[:, None]
        # Parameters from Eqs 21, 22
        a1 = [0.857, 0.153, 1.840, 0.254]
        a2 = [0.691, 1.330, 1.668, 0.534]
        z = gamma_energy / electron_energy
        ttheta = (
            2.0
            * electron_energy
            * soft_photon_temperature
            * (1.0 - np.cos(theta))
        )
        x = z / (1 - z) / ttheta
        # Eq. 11
        cross_section = z ** 2 / (2 * (1 - z)) * G12(x, a1) + G12(x, a2)
        tmp = (soft_photon_temperature / electron_energy) ** 2
        # r0 = (e**2 / m_e / c**2).to('cm')
        # (2 * r0 ** 2 * m_e ** 3 * c ** 4 / (pi * hbar ** 3)).cgs
        tmp *= 2.6318735743809104e16
        cross_section = tmp * cross_section
        cc = (gamma_energy < electron_energy) * (electron_energy > 1)
        return np.where(cc, cross_section, np.zeros_like(cross_section))

    @staticmethod
    def _iso_ic_on_monochromatic(
        electron_energy, seed_energy, seed_edensity, gamma_energy
    ):
        """
        IC cross-section for an isotropic interaction with a monochromatic
        photon spectrum following Eq. 22 of Aharonian & Atoyan 1981, Ap&SS 79,
        321 (`http://adsabs.harvard.edu/abs/1981Ap%26SS..79..321A`_)
        """
        photE0 = (seed_energy / mec2).decompose().value
        phn = seed_edensity

        # electron_energy = electron_energy[:, None]
        gamma_energy = gamma_energy[:, None]
        photE0 = photE0[:, None, None]
        phn = phn[:, None, None]

        b = 4 * photE0 * electron_energy
        w = gamma_energy / electron_energy
        q = w / (b * (1 - w))
        fic = (
            2 * q * np.log(q)
            + (1 + 2 * q) * (1 - q)
            + (1.0 / 2.0) * (b * q) ** 2 * (1 - q) / (1 + b * q)
        )

        gamint = (
            fic
            * heaviside(1 - q)
            * heaviside(q - 1.0 / (4 * electron_energy ** 2))
        )
        gamint[np.isnan(gamint)] = 0.0

        if phn.size > 1:
            phn = phn.to(1 / (mec2_unit * u.cm ** 3)).value
            gamint = trapz_loglog(gamint * phn / photE0, photE0, axis=0)  # 1/s
        else:
            phn = phn.to(mec2_unit / u.cm ** 3).value
            gamint *= phn / photE0 ** 2
            gamint = gamint.squeeze()

        # gamint /= mec2.to('erg').value

        # r0 = (e**2 / m_e / c**2).to('cm')
        # sigt = ((8 * np.pi) / 3 * r0**2).cgs
        sigt = 6.652458734983284e-25
        c = 29979245800.0

        gamint *= (3.0 / 4.0) * sigt * c / electron_energy ** 2

        return gamint

    def _calc_specic(self, seed, outspecene):
        # log.debug(
        #     "_calc_specic: Computing IC on {0} seed photons...".format(seed)
        # )

        Eph = (outspecene / mec2).decompose().value
        # Catch numpy RuntimeWarnings of overflowing exp (which are then
        # discarded anyway)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.seed_photon_fields[seed]["type"] == "thermal":
                T = self.seed_photon_fields[seed]["T"]
                uf = (
                    self.seed_photon_fields[seed]["u"] / (ar * T ** 4)
                ).decompose()
                if self.seed_photon_fields[seed]["isotropic"]:
                    gamint = self._iso_ic_on_planck(
                        self._gam, T.to("K").value, Eph
                    )
                else:
                    theta = (
                        self.seed_photon_fields[seed]["theta"].to("rad").value
                    )
                    gamint = self._ani_ic_on_planck(
                        self._gam, T.to("K").value, Eph, theta
                    )
            else:
                uf = 1
                gamint = self._iso_ic_on_monochromatic(
                    self._gam,
                    self.seed_photon_fields[seed]["energy"],
                    self.seed_photon_fields[seed]["photon_density"],
                    Eph,
                )

            lum = uf * Eph * trapz_loglog(self._nelec * gamint, self._gam)
        lum = lum * u.Unit("1/s")

        return lum / outspecene  # return differential spectrum in 1/s/eV

    def _spectrum(self, photon_energy):
        """Compute differential IC spectrum for energies in ``photon_energy``.

        Compute IC spectrum using IC cross-section for isotropic interaction
        with a blackbody photon spectrum following Khangulyan, Aharonian, and
        Kelner 2014, ApJ 783, 100 (`arXiv:1310.7971
        <http://www.arxiv.org/abs/1310.7971>`_).

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` instance
            Photon energy array.
        """
        outspecene = _validate_ene(photon_energy)

        self.specic = []

        for seed in self.seed_photon_fields:
            # Call actual computation, detached to allow changes in subclasses
            self.specic.append(
                self._calc_specic(seed, outspecene).to("1/(s eV)")
            )

        return np.sum(u.Quantity(self.specic), axis=0)

    def flux(self, photon_energy, distance=1 * u.kpc, seed=None):
        """Differential flux at a given distance from the source from a single
        seed photon field

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.

        seed : int, str or None
            Number or name of seed photon field for which the IC contribution
            is required. If set to None it will return the sum of all
            contributions (default).
        """
        model = super().flux(photon_energy, distance=distance)

        if seed is not None:
            # Test seed argument
            if not isinstance(seed, int):
                if seed not in self.seed_photon_fields:
                    raise ValueError(
                        "Provided seed photon field name is not in"
                        " the definition of the InverseCompton instance"
                    )
                else:
                    seed = list(self.seed_photon_fields.keys()).index(seed)
            elif seed > len(self.seed_photon_fields):
                raise ValueError(
                    "Provided seed photon field number is larger"
                    " than the number of seed photon fields defined in the"
                    " InverseCompton instance"
                )

            if distance != 0:
                distance = validate_scalar(
                    "distance", distance, physical_type="length"
                )
                dfac = 4 * np.pi * distance.to("cm") ** 2
                out_unit = "1/(s cm2 eV)"
            else:
                dfac = 1
                out_unit = "1/(s eV)"

            model = (self.specic[seed] / dfac).to(out_unit)

        return model


    def sed(self, photon_energy, distance=1 * u.kpc, seed=None):
        """Spectral energy distribution at a given distance from the source

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.

        seed : int, str or None
            Number or name of seed photon field for which the IC contribution
            is required. If set to None it will return the sum of all
            contributions (default).
        """
        sed = super().sed(photon_energy, distance=distance)

        if seed is not None:
            if distance != 0:
                out_unit = "erg/(cm2 s)"
            else:
                out_unit = "erg/s"

            sed = (
                self.flux(photon_energy, distance=distance, seed=seed)
                * photon_energy ** 2.0
            ).to(out_unit)

        return sed


class TableModel:
    """
    A model generated from a table of energy and value arrays.

    The units returned will be the units of the values array provided at
    initialization. The model will return values interpolated in
    log-space, returning 0 for energies outside of the limits of the provided
    energy array.

    Parameters
    ----------
    energy : `~astropy.units.Quantity` array
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    amplitude : float
        Model amplitude that is multiplied to the supplied arrays. Defaults to
        1.
    """

    def __init__(self, energy, values, amplitude=1):
        from scipy.interpolate import interp1d

        self._energy = validate_array(
            "energy", energy, domain="positive", physical_type="energy"
        )
        self._values = values
        self.amplitude = amplitude

        loge = np.log10(self._energy.to("eV").value)
        try:
            self.unit = self._values.unit
            logy = np.log10(self._values.value)
        except AttributeError:
            self.unit = u.Unit("")
            logy = np.log10(self._values)

        self._interplogy = interp1d(
            loge, logy, fill_value=-np.Inf, bounds_error=False, kind="cubic"
        )

    def __call__(self, e):
        e = _validate_ene(e)
        interpy = np.power(10, self._interplogy(np.log10(e.to("eV").value)))
        return self.amplitude * interpy * self.unit


class EblAbsorptionModel(TableModel):
    """
    A TableModel containing the different absorption values from a specific
    model.

    It returns dimensionless opacity values, that could be multiplied to any
    model.

    Parameters
    ----------
    redshift : float
        Redshift considered for the absorption evaluation.
    ebl_absorption_model : {'Dominguez'}
        Name of the EBL absorption model to use (Dominguez by default).

    Notes
    -----
    Dominguez model refers to the Dominguez 2011 EBL model. Current
    implementation does NOT perform an interpolation in the redshift, so it
    just uses the closest z value from the finely binned tau_dominguez11.npz
    file (delta_z=0.01).

    See Also
    --------
    TableModel
    """

    def __init__(self, redshift, ebl_absorption_model="Dominguez"):

        # check that the redshift is a positive scalar
        if not isinstance(redshift, u.Quantity):
            redshift *= u.dimensionless_unscaled

        self.redshift = validate_scalar(
            "redshift",
            redshift,
            domain="positive",
            physical_type="dimensionless",
        )

        self.model = ebl_absorption_model

        if self.model == "Dominguez":
            """Table generated by Alberto Dominguez containing tau vs energy
            [TeV] vs redshift.  Energy is defined between 1 GeV and 100 TeV, in
            500 bins uniform in log(E).  Redshift is defined between 0.01 and
            4, in steps of 0.01."""
            filename = get_pkg_data_filename(
                os.path.join("data", "ebl", "tau_dominguez11.npz")
            )
            taus_table = np.load(filename)["arr_0"]
            redshift_list = np.arange(0.01, 4, 0.01)
            energy = taus_table["energy"] * u.TeV
            if self.redshift >= 0.01:
                colname = "col%s" % (
                    2 + (np.abs(redshift_list - self.redshift)).argmin()
                )
                table_values = taus_table[colname]
                # Set maximum value of the log(Tau) to 150, as it is high
                # enough.  This solves later overflow problems.
                table_values[table_values > 150.0] = 150.0
                taus = 10 ** table_values * u.dimensionless_unscaled
            elif self.redshift < 0.01:
                taus = (
                    10 ** np.zeros(len(taus_table["energy"]))
                    * u.dimensionless_unscaled
                )
        else:
            raise ValueError('Model should be one of: ["Dominguez"]')

        super().__init__(energy, taus)

    def transmission(self, e):
        e = _validate_ene(e)
        taus = np.zeros(len(e))
        for i in range(0, len(e)):
            if e[i].to("GeV").value < 1.0:
                taus[i] = 0.0
            elif e[i].to("TeV").value > 100.0:
                taus[i] = np.log10(6000.0)
            else:
                taus[i] = np.log10(self(e[i]))
        return np.exp(-taus)
