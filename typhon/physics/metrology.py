"""Functions related to metrology

All functions need sympy.
"""


def express_uncertainty(expr, aliases={}):
    """For a sympy expression, calculate uncertainty.

    Uncertainty is given in the Guides to Uncertainties and Measurements
    (GUM), see http://www.bipm.org/en/publications/guides/gum.html
    equations 10 and 13.

    This takes all free symbols.  If you have IndexedBase symbols, you may
    want to pass them into aliases {free_symbol:
    indexed_base_symbol[index]} is this is not identified automatically.

    Limitation: currently assumes all input quantities are uncorrelated!

    Arguments:
        
        expr (Expr): Expression for which to calculate uncertainty
        aliases (Mapping): Mapping of replacements to apply prior.

    Returns:
        
        Expression indicating uncertainty
    """

    import sympy
    u = sympy.Function("u")
    rv = sympy.sympify(0)
    for sym in recursive_args(expr):
        sym = aliases.get(sym, sym)
        rv += sympy.diff(expr, sym)**2 * u(sym)**2
    return sympy.sqrt(rv)

def recursive_args(expr, stop_at=None, partial_at=None):
    """Get arguments for `expr`, stopping at certain types

    Get all arguments in expression down to the levels in `expr`.  When
    `expr` is only `{sympy.Symbol}` this is identical to
    `expr.free_symbols`, but in some cases we want to retain IndexedBase,
    for example, when evaluating uncertainies.

    This is mainly a helper for express_uncertainty where we don't want to
    descend beyond Indexed quantities
    """

    import sympy
    import sympy.concrete.expr_with_limits
    if stop_at is None:
        stop_at = (sympy.Symbol, sympy.Indexed)
    if partial_at is None:
        partial_at = sympy.concrete.expr_with_limits.ExprWithLimits
    args = set()
    if isinstance(expr, stop_at):
        return args
    if isinstance(expr, partial_at):
        return recursive_args(expr.args[0])
    for arg in expr.args:
        if isinstance(arg, stop_at):
            args.add(arg)
        elif isinstance(arg, partial_at):
            args.update(recursive_args(arg.args[0], stop_at=stop_at))
        else:
            args.update(recursive_args(arg, stop_at=stop_at))
    return args
