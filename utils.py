
# from __future__ import (
#     absolute_import,
#     division,
#     print_function,
#     unicode_literals,
# )

import warnings
import hashlib
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, dblquad
from scipy.stats import truncnorm


def heaviside(x):
    return (np.sign(x) + 1) / 2.0


def get_truncnorm(num, mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(num)


def cos_sph(theta_1, phi_1, theta_2, phi_2):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    return np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2) + np.cos(
        theta_1) * np.cos(theta_2)


def memoize(func):
    """ Cache decorator for functions inside model classes """

    def model(cls, energy, *args, **kwargs):
        try:
            memoize = cls._memoize
            cache = cls._cache
            queue = cls._queue
        except AttributeError:
            memoize = False

        if memoize:
            # Allow for dicts or tables with energy column, Quantity array or
            # Quantity scalar
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore",
                        getattr(np, "VisibleDeprecationWarning", None),
                    )
                    energy = u.Quantity(energy["energy"])
            except (TypeError, ValueError, IndexError):
                pass

            try:
                # tostring is 10 times faster than str(array).encode()
                bstr = energy.value.tostring()
            except AttributeError:
                # scalar Quantity
                bstr = str(energy.value).encode()

            data = [hashlib.sha256(bstr).hexdigest()]

            data.append(energy.unit.to_string())
            data.append(str(kwargs.get("distance", 0)))

            if args:
                data.append(str(args))
            if hasattr(cls, "particle_distribution"):
                models = [cls, cls.particle_distribution]
            else:
                models = [cls]
            for model in models:
                if hasattr(model, "param_names"):
                    for par in model.param_names:
                        data.append(str(getattr(model, par)))

            token = "".join(data)
            digest = hashlib.sha256(token.encode()).hexdigest()

            if digest in cache:
                return cache[digest]

        result = func(cls, energy, *args, **kwargs)

        if memoize:
            # remove first item in queue and remove from cache
            if len(queue) > 16:
                key = queue.pop(0)
                cache.pop(key, None)
            # save last result to cache
            queue.append(digest)
            cache[digest] = result

        return result

    model.__name__ = func.__name__
    model.__doc__ = func.__doc__

    return model


def validate_physical_type(name, value, physical_type):
    if physical_type is not None:
        if not isinstance(value, u.Quantity):
            raise TypeError(
                "{0} should be given as a Quantity object".format(name)
            )
        if isinstance(physical_type, str):
            if value.unit.physical_type != physical_type:
                raise TypeError(
                    "{0} should be given in units of {1}".format(
                        name, physical_type
                    )
                )
        else:
            if not value.unit.physical_type in physical_type:
                raise TypeError(
                    "{0} should be given in units of {1}".format(
                        name, ", ".join(physical_type)
                    )
                )


def validate_scalar(name, value, domain=None, physical_type=None):

    validate_physical_type(name, value, physical_type)

    if not physical_type:
        if not np.isscalar(value) or not np.isreal(value):
            raise TypeError(
                "{0} should be a scalar floating point value".format(name)
            )

    if domain == "positive":
        if value < 0.0:
            raise ValueError("{0} should be positive".format(name))
    elif domain == "strictly-positive":
        if value <= 0.0:
            raise ValueError("{0} should be strictly positive".format(name))
    elif domain == "negative":
        if value > 0.0:
            raise ValueError("{0} should be negative".format(name))
    elif domain == "strictly-negative":
        if value >= 0.0:
            raise ValueError("{0} should be strictly negative".format(name))
    elif type(domain) in [tuple, list] and len(domain) == 2:
        if value < domain[0] or value > domain[-1]:
            raise ValueError(
                "{0} should be in the range [{1}:{2}]".format(
                    name, domain[0], domain[-1]
                )
            )

    return value


def validate_array(
    name, value, domain=None, ndim=1, shape=None, physical_type=None
):

    validate_physical_type(name, value, physical_type)

    # First convert to a Numpy array:
    if type(value) in [list, tuple]:
        value = np.array(value)

    # Check the value is an array with the right number of dimensions
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        if ndim == 1:
            raise TypeError("{0} should be a 1-d sequence".format(name))
        else:
            raise TypeError("{0} should be a {1:d}-d array".format(name, ndim))

    # Check that the shape matches that expected
    if shape is not None and value.shape != shape:
        if ndim == 1:
            raise ValueError(
                "{0} has incorrect length (expected {1} but found {2})".format(
                    name, shape[0], value.shape[0]
                )
            )
        else:
            raise ValueError(
                "{0} has incorrect shape (expected {1} but found {2})".format(
                    name, shape, value.shape
                )
            )

    return value


def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y = np.where(x<xp[0], yp[0]+(x-xp[0])*(yp[0]-yp[1])/(xp[0]-xp[1]), y)
    y = np.where(x>xp[-1], yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]), y)
    return y


def extrap_loglog(x, xp, fp):
    return 10**(extrap(np.log10(x), np.log10(xp), np.log10(fp)))


def _value(q):
    try:
        return q.value
    except AttributeError:
        return q


def _unit(q):
    try:
        return q.unit
    except AttributeError:
        # must not return u.dimensionless_unscaled to avoid
        # UnitTypeError: Can only apply 'sin' function to quantities with angle units
        return 1.


def _to_value(q, unit=None):
    if unit == None:
        unit = _unit(q)
    return (q * u.dimensionless_unscaled).to(unit).value


def traverse(data, tree_types=(list, tuple, np.ndarray)):
    if isinstance(data, tree_types) and np.ndim(np.array(data, dtype=object)) > 0:
        for value in data:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield data

# np.atleast_1d
def unit_extractor(q, q_unit=None):
    if q_unit is None:
        try:
            q_unit = _unit(next(traverse(q)))
        except StopIteration:
            return np.array(q)
    q_shape = np.array(q, dtype=object).shape
    return np.array([_to_value(_q, q_unit)
                     for _q in traverse(q)]).reshape(q_shape) * q_unit


def qextrap_loglog(x, xp, fp):
    return extrap_loglog(_to_value(x, _unit(xp)), _value(xp), _value(fp)) * _unit(fp)


def qinterp1d(x, y, **kwargs):
    return lambda xx: interp1d(_value(x), _value(y), **kwargs)(
        _to_value(xx, _unit(x))) * _unit(y)


def qinterp1d_loglog(x, y, **kwargs):
    return lambda xx: 10**(interp1d(np.log10(_value(x)), np.log10(_value(y)), **kwargs)(
        np.log10(_to_value(xx, _unit(x))))) * _unit(y)


def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (
                y[slice1]
                * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1])
            )
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret


def qquad(func, a, b, **kwargs):
    x_unit = _unit(a)
    y_unit = _unit(func(a))

    a = _to_value(a, x_unit)
    b = _to_value(b, x_unit)

    func_dimless = lambda x: _to_value(func(x*x_unit), y_unit)
    unit = y_unit * x_unit
    return quad(func_dimless, a, b, **kwargs) * unit


def dblqquad(func, a, b, gfun, hfun, **kwargs):
    x_unit = _unit(a)

    gfun_test = gfun(a) if callable(gfun) else gfun
    y_unit = _unit(gfun_test)

    z_unit = _unit(func(gfun_test, a))

    a = _to_value(a, x_unit)
    b = _to_value(b, x_unit)

    if callable(gfun):
        gfun_dimless = lambda x: _to_value(gfun(x*x_unit), y_unit)
    else:
        gfun_dimless = _to_value(gfun, y_unit)
    if callable(hfun):
        hfun_dimless = lambda x: _to_value(hfun(x*x_unit), y_unit)
    else:
        hfun_dimless = _to_value(hfun, y_unit)

    func_dimless = lambda y, x: _to_value(func(y*y_unit, x*x_unit), z_unit)

    unit = z_unit * x_unit * y_unit
    return dblquad(func_dimless, a, b, gfun_dimless, hfun_dimless, **kwargs) * unit


def quad_loglog(func, a, b, **kwargs):
    return quad(lambda logx: np.exp(logx) * func(np.exp(logx)), np.log(a),
                np.log(b), **kwargs)


def qquad_loglog(func, a, b, **kwargs):
    x_unit = _unit(a)
    y_unit = _unit(func(a))

    a = _to_value(a, x_unit)
    b = _to_value(b, x_unit)

    func_dimless = lambda x: _to_value(func(x*x_unit), y_unit)
    unit = y_unit * x_unit

    return quad(lambda logx: np.exp(logx) * func_dimless(np.exp(logx)), np.log(a),
                np.log(b), **kwargs) * unit


def trapz_sphavg(dfdOmega, theta_l=0, theta_r=np.pi, phi_h=2*np.pi, theta_step=8, phi_step=3):
    theta_arr = np.linspace(theta_l, theta_r, theta_step)
    phi_arr = np.linspace(-phi_h/2, phi_h/2, phi_step)
    return np.trapz(unit_extractor([
        np.trapz(unit_extractor(
            [dfdOmega(theta, phi)*np.sin(theta) / 4 / np.pi
            for phi in phi_arr]).T, phi_arr) 
            for theta in theta_arr]).T, theta_arr)


def qquad_sphavg(dfdOmega, theta_l=0, theta_r=np.pi, phi_h=2*np.pi):

    def dfdOmega_sin(theta, phi):
        # np.sin(0*u.dimensionless_unscaled)
        # UnitTypeError: Can only apply 'sin' function to quantities with angle units
        return dfdOmega(theta, phi) * np.sin(theta) / 4 / np.pi

    return dblqquad(dfdOmega_sin, -phi_h/2, phi_h/2, theta_l, theta_r)


class extrap2d(interp2d):
    def __init__(self, x, y, z, kind='linear'):
        if kind == 'linear':
            if len(x) < 2 or len(y) < 2:
                raise self.get_size_error(2, kind)
        elif kind == 'cubic':
            if len(x) < 4 or len(y) < 4:
                raise self.get_size_error(4, kind)
        elif kind == 'quintic':
            if len(x) < 6 or len(y) < 6:
                raise self.get_size_error(6, kind)
        else:
            raise ValueError('unidentifiable kind of spline')

        super().__init__(x, y, z, kind=kind)
        self.kind = kind
        self.extrap_fd_based_xs = self._linspace_10(self.x_min, self.x_max, -4)
        self.extrap_bd_based_xs = self._linspace_10(self.x_min, self.x_max, 4)
        self.extrap_fd_based_ys = self._linspace_10(self.y_min, self.y_max, -4)
        self.extrap_bd_based_ys = self._linspace_10(self.y_min, self.y_max, 4)

    @staticmethod
    def get_size_error(size, spline_kind):
        return ValueError('length of x and y must be larger or at least equal '
                          'to {} when applying {} spline, assign arrays with '
                          'length no less than '
                          '{}'.format(size, spline_kind, size))

    @staticmethod
    def _extrap1d(xs, ys, tar_x, kind):
        if isinstance(xs, np.ndarray):
            xs = np.ndarray.flatten(xs)
        if isinstance(ys, np.ndarray):
            ys = np.ndarray.flatten(ys)
        assert len(xs) >= 4
        assert len(xs) == len(ys)
        # f = InterpolatedUnivariateSpline(xs, ys)
        f = interp1d(xs, ys, kind=kind, fill_value='extrapolate')
        return f(tar_x)

    @staticmethod
    def _linspace_10(p1, p2, cut=None):
        ls = list(np.linspace(p1, p2, 10))
        if cut is None:
            return ls
        assert cut <= 10
        return ls[-cut:] if cut < 0 else ls[:cut]

    def _get_extrap_based_points(self, axis, extrap_p):
        if axis == 'x':
            return (self.extrap_fd_based_xs if extrap_p > self.x_max else
                    self.extrap_bd_based_xs if extrap_p < self.x_min else [])
        elif axis == 'y':
            return (self.extrap_fd_based_ys if extrap_p > self.y_max else
                    self.extrap_bd_based_ys if extrap_p < self.y_min else [])
        assert False, 'axis unknown'
        
    def __call__(self, x_, y_, **kwargs):
        xs = np.atleast_1d(x_)
        ys = np.atleast_1d(y_)

        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")

        pz_yqueue = []
        for y in ys:
            extrap_based_ys = self._get_extrap_based_points('y', y)

            pz_xqueue = []
            for x in xs:
                extrap_based_xs = self._get_extrap_based_points('x', x)

                if not extrap_based_xs and not extrap_based_ys:
                    # inbounds
                    pz = super().__call__(x, y, **kwargs)[0]

                elif extrap_based_xs and extrap_based_ys:
                    # both x, y atr outbounds
                    # allocate based_z from x, based_ys
                    extrap_based_zs = self.__call__(x,
                                                    extrap_based_ys,
                                                    **kwargs)
                    # allocate z of x, y from based_ys, based_zs
                    pz = self._extrap1d(extrap_based_ys, extrap_based_zs, y, self.kind)

                elif extrap_based_xs:
                    # only x outbounds
                    extrap_based_zs = super().__call__(extrap_based_xs,
                                                       y,
                                                       **kwargs)
                    pz = self._extrap1d(extrap_based_xs, extrap_based_zs, x, self.kind)

                else:
                    # only y outbounds
                    extrap_based_zs = super().__call__(x,
                                                       extrap_based_ys,
                                                       **kwargs)
                    pz = self._extrap1d(extrap_based_ys, extrap_based_zs, y, self.kind)

                pz_xqueue.append(pz)

            pz_yqueue.append(pz_xqueue)

        zss = pz_yqueue
        if len(zss) == 1:
            zss = zss[0]
        return np.array(zss)