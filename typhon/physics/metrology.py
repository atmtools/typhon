"""Functions related to metrology

This module has a soft dependency on sympy <http://www.sympy.org>
"""

def express_uncertainty(expr):
    """For a sympy expression, calculate uncertainty.

    Uncertainty is given in the Guides to Uncertainties and Measurements
    (GUM), see http://www.bipm.org/en/publications/guides/gum.html
    equations 10 and 13.

    Limitation: currently assumes all input quantities are uncorrelated!

    Arguments:
        
        expr (Expr): Expression for which to calculate uncertainty

    Returns:
        
        Expression indicating uncertainty
    """

    import sympy
    u = sympy.Function("u")
    rv = sympy.sympify(0)
    for sym in expr.free_symbols:
        rv += sympy.diff(expr, sym)**2 * u(sym)**2
    return sympy.sqrt(rv)
