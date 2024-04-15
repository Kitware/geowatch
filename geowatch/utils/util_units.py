import ubelt as ub


@ub.memoize
def unit_registry():
    """
    A memoized unit registry
    """
    import pint
    ureg = pint.UnitRegistry()
    return ureg
