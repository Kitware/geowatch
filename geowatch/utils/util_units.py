import ubelt as ub


@ub.memoize
def unit_registry():
    """
    A memoized unit registery
    """
    import pint
    ureg = pint.UnitRegistry()
    return ureg
