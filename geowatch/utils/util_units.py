import ubelt as ub


@ub.memoize
def unit_registery():
    """
    A memoized unit registery
    """
    import pint
    ureg = pint.UnitRegistry()
    return ureg
