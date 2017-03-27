"""Miscellaneous unit-aware tools
"""

from .common import ureg

import operator
import numpy
import xarray
class UnitsAwareDataArray(xarray.DataArray):
    """Like xarray.DataArray, but transfers units
    """

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
                new_var.attrs["units"] = str(u)
                #new_var.attrs["units"] = context[0](ureg(self.attrs.get("units"))).u
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

    def to(self, new_unit, *contexts, **kwargs):
        """Convert to other unit.

        See corresponding method in pint.
        """
        x = self.copy()
        x.values = ureg.Quantity(self.values, self.attrs["units"]).to(
            new_unit, *contexts, **kwargs).m
        x.attrs["units"] = str(new_unit)
        return x

tp_all = ["add", "sub", "mul", "matmul", "truediv", "floordiv", "mod",
          "divmod"]
for tp in tp_all + ["r"+x for x in tp_all]:
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        x = getattr(super(UnitsAwareDataArray, self), meth)(other)
        if tp.startswith("r"):
            return self._apply_rbinary_op_to_units(getattr(operator, tp[1:]), other, x)
        else:
            return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitsAwareDataArray, meth, func)
del func, tp_all
