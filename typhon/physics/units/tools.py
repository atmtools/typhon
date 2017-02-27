"""Miscellaneous unit-aware tools
"""

from .common import ureg

import operator
import xarray
class UnitsAwareDataArray(xarray.DataArray):
    """Like xarray.DataArray, but transfers units
    """

    def __array_wrap__(self, obj, context=None):
        new_var = super().__array_wrap__(obj, context)
        if self.attrs.get("units"):
            if context:
                new_var.attrs["units"] = context[0](ureg(self.attrs.get("units"))).u
            else: # unary operators always retain units?
                new_var.attrs["units"] = self.attrs.get("units")
        return new_var

    def _apply_binary_op_to_units(self, func, other, x):
        if self.attrs.get("units"):
            x.attrs["units"] = func(ureg.Quantity(1, self.attrs["units"]),
                                    ureg.Quantity(1, getattr(other, "units", "1"))).u
        return x

    # pow is different because resulting unit depends on argument, not on
    # unit of argument
    def __pow__(self, other):
        x = super().__pow__(other)
        if self.attrs.get("units"):
            x.attrs["units"] = pow(ureg.Quantity(1, self.attrs["units"]),
                                   ureg.Quantity(other, getattr(other, "units", "1"))).u
        return x

    def to(self, new_unit, context=None, **kwargs):
        """Convert to other unit.

        See corresponding method in pint.
        """
        x = self.copy()
        x.values = ureg.Quantity(self.values, self.attrs["units"]).to(
            new_unit, context, **kwargs).m
        x.attrs["units"] = new_unit
        return x


for tp in ("add", "sub", "mul", "matmul", "truediv", "floordiv", "mod",
    "divmod"):
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        x = getattr(super(UnitsAwareDataArray, self), meth)(other)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitsAwareDataArray, meth, func)
del func
