
def fix_sorted_set():
    # (kwcoco 0.5.1 renamed SortedSetQuiet to SortedSet)
    import kwcoco
    if('SortedSetQuiet' not in dir(kwcoco._helpers)  # noqa: E275
       and 'SortedSet' in dir(kwcoco._helpers)):
        kwcoco._helpers.SortedSetQuiet = kwcoco._helpers.SortedSet
