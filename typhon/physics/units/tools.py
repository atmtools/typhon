"""Miscellaneous unit-aware tools
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

from .common import ureg

import operator
import numpy
import xarray
class UnitsAwareDataArray(xarray.DataArray):
    """Like xarray.DataArray, but transfers units
    """

    # need to keep both __array_wrap__ and __array_ufunc__.  Although the
    # former supersedes the latter, xarrays methods explicitly call the
    # former sometimes.
    def __array_wrap__(self, obj, context=None):
        new_var = super().__array_wrap__(obj, context)
        if self.attrs.get("units"):
            if context:
                # some ufuncs like exp return regular numpy floats rather
                # than (dimensionless) quantities, see
                # https://github.com/hgrecco/pint/issues/482
                q = context[0](ureg.Quantity(1, self.attrs.get("units")))
                try:
                    u = q.u
                except AttributeError:
                    if (ureg(self.attrs["units"]).dimensionless or
                        new_var.dtype.kind == "b"):
                        # expected, see # https://github.com/hgrecco/pint/issues/482
                        u = ureg.dimensionless
                    else:
                        raise
                    # for exp and log, values are not set correctly.  I'm
                    # not sure why.  Perhaps related to
                    # https://github.com/hgrecco/pint/issues/493
                    new_var.values = context[0](ureg.Quantity(self.values, self.units))
                new_var.attrs["units"] = str(u)
            else: # unary operators always retain units?
                new_var.attrs["units"] = str(self.attrs.get("units"))
        return new_var

    def _apply_binary_op_to_units(self, func, other, x):
        if self.attrs.get("units"):
            x.attrs["units"] = str(func(ureg.Quantity(1, self.attrs["units"]),
                                    ureg.Quantity(1, getattr(other, "units", "1"))).u)
        return x

    def _apply_rbinary_op_to_units(self, func, other, x):
        if self.attrs.get("units"):
            x.attrs["units"] = str(func(ureg.Quantity(1, getattr(other, "units", "1")),
                                    ureg.Quantity(1, self.attrs["units"]),).u)
        return x

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        new_var = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        # make sure we're still UADA
        new_var = self.__class__(new_var)
        if self.attrs.get("units"):
            if method == "__call__":
                q = ufunc(ureg.Quantity(1, self.attrs.get("units")))
                try:
                    u = q.u
                except AttributeError:
                    if (ureg(self.attrs["units"]).dimensionless or
                        new_var.dtype.kind == "b"):
                        # expected, see https://github.com/hgrecco/pint/issues/482
                        u = ureg.dimensionless
                    else:
                        raise
                    # for exp and log, values are not set correctly.  I'm
                    # not sure why.  Perhaps related to
                    # https://github.com/hgrecco/pint/issues/493
                    new_var.values = ufunc(ureg.Quantity(self.values, self.units))
                new_var.attrs["units"] = str(u)
            else: # unary operators? always retain units?
                raise NotImplementedError("Not implented")
                new_var.attrs["units"] = str(self.attrs.get("units"))
                    
        return new_var

    # pow is different because resulting unit depends on argument, not on
    # unit of argument
    def __pow__(self, other):
        x = super().__pow__(other)
        if self.attrs.get("units"):
            x.attrs["units"] = str(pow(ureg.Quantity(1, self.attrs["units"]),
                                   ureg.Quantity(other, getattr(other, "units", "1"))).u)
        return x

    def __rpow__(self, other):
        x = super().__rpow__(other)
        if self.attrs.get("units"):
            # NB: this should fail if we're not dimensionless
            x.attrs["units"] = str(pow(
                ureg.Quantity(other, getattr(other, "units", "1")),
                ureg.Quantity(1, self.attrs["units"])).u)
        return x

    def _reduce(self, attr, *args, **kwargs):
        # unit on diff, sum etc. is a bit involved as it should depend on the
        # coordinates of the dimension along which the reduction is taken, but
        # as it is all discrete I consider the unit to stay the same
        u = self.attrs.get("units")
        d = getattr(super(), attr)(*args, **kwargs)
        if u is not None:
            d.attrs["units"] = u
        return d

    # not sure if/how I can/should define those in a loop.  Should also have
    # std, min, max, mean, probably others.
    def diff(self, *args, **kwargs):
        return self._reduce("diff", *args, **kwargs)

    def sum(self, *args, **kwargs):
        return self._reduce("sum", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._reduce("mean", *args, **kwargs)

    def median(self, *args, **kwargs):
        return self._reduce("median", *args, **kwargs)

    def to(self, new_unit, *contexts, **kwargs):
        """Convert to other unit.

        See corresponding method in pint.
        """
        x = self.copy()
        x.values = ureg.Quantity(self.values, self.attrs["units"]).to(
            new_unit, *contexts, **kwargs).m
        x.attrs["units"] = str(new_unit)
        return x

    def to_root_units(self):
        """Equivalent of pint's Quantity.to_root_units
        """
        x = self.copy()
        q = ureg.Quantity(self.values, self.attrs["units"]).to_root_units()
        x.values = q.m
        x.attrs["units"] = str(q.u)
        return x

tp_all = ["add", "sub", "mul", "matmul", "truediv", "floordiv", "mod",
          "divmod"]
for tp in tp_all + ["r"+x for x in tp_all]:
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        rat = (ureg.Quantity(1, getattr(self, "units", "1")) /
               ureg.Quantity(1, getattr(other, "units", "1")))
        if (rat.u.dimensionless and rat.m != 1
#        if (ureg.Quantity(1, getattr(self, "units", "1")) /
#            ureg.Quantity(1, getattr(other, "units", "1")))
#        if (ureg.Quantity(1, getattr(self, "units", "1")).to_root_units()
#            == ureg.Quantity(1, getattr(other, "units", "1")).to_root_units()
#            and ureg.Quantity(1, self.units).m != ureg.Quantity(1, self.units).to("1").m
#            and ureg.Quantity(1, self.units).m != ureg.Quantity(1, self.units).to("1").m
#            and getattr(self, "units", "1") != getattr(other, "units", "1")
            and meth[2:-2] not in ("mul", "rmul", "floordiv", "truediv")):
            raise NotImplementedError("Do not add arrays with same units "
                "but different prefixes, currently buggy, see #150.")
        x = getattr(super(UnitsAwareDataArray, self), meth)(other)
        if tp.startswith("r"):
            return self._apply_rbinary_op_to_units(getattr(operator, tp[1:]), other, x)
        else:
            return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitsAwareDataArray, meth, func)
del func, tp_all
