import string


def partial_format(format_string, *args, **kwargs):
    """
    A solution to the partial string formatting problem.

    Taken from [SO11283961]_, which is a modification of the stdlib
    string.Formatter code.

    Args:
        format_string (str): the templated string to be formatted
        *args: positional replacements
        **kwargs: key value replacements

    Returns:
        str : the format string with only the specified parts replaced.

    Example:
        >>> from watch.utils.partial_format import partial_format
        >>> format_string = '{foo} + {bar} = {baz}'
        >>> args = tuple()
        >>> kwargs = dict(bar=3)
        >>> partial_format(format_string, *args, **kwargs)
        '{foo} + 3 = {baz}'

    References:
        [SO11283961] https://stackoverflow.com/questions/11283961/partial-string-formatting
    """
    return _PartialFormatter().format(format_string, *args, **kwargs)


class _PartialFormatter(string.Formatter):
    """
    A modified string formatter that handles a partial set of format
    args/kwargs.
    """

    def vformat(self, format_string, args, kwargs):
        used_args = set()
        result, _ = self._vformat(format_string, args, kwargs, used_args, 2)
        self.check_unused_args(used_args, args, kwargs)
        return result

    def _vformat(self, format_string, args, kwargs, used_args, recursion_depth,
                 auto_arg_index=0):
        if recursion_depth < 0:
            raise ValueError('Max string recursion exceeded')
        result = []
        for literal_text, field_name, format_spec, conversion in \
                self.parse(format_string):

            orig_field_name = field_name

            # output the literal text
            if literal_text:
                result.append(literal_text)

            # if there's a field, output it
            if field_name is not None:
                # this is some markup, find the object and do
                #  the formatting

                # handle arg indexing when empty field_names are given.
                if field_name == '':
                    if auto_arg_index is False:
                        raise ValueError('cannot switch from manual field '
                                         'specification to automatic field '
                                         'numbering')
                    field_name = str(auto_arg_index)
                    auto_arg_index += 1
                elif field_name.isdigit():
                    if auto_arg_index:
                        raise ValueError('cannot switch from manual field '
                                         'specification to automatic field '
                                         'numbering')
                    # disable auto arg incrementing, if it gets
                    # used later on, then an exception will be raised
                    auto_arg_index = False

                # given the field_name, find the object it references
                #  and the argument it came from
                try:
                    obj, arg_used = self.get_field(field_name, args, kwargs)
                except (IndexError, KeyError):
                    ##########################
                    # Where the magic happens.
                    # ------------------------
                    # This case is the main difference between this class and
                    # the stdlib implementation.
                    ##########################
                    # catch issues with both arg indexing and kwarg key errors
                    obj = orig_field_name
                    if conversion:
                        obj += '!{}'.format(conversion)
                    if format_spec:
                        format_spec, auto_arg_index = self._vformat(
                            format_spec, args, kwargs, used_args,
                            recursion_depth, auto_arg_index=auto_arg_index)
                        obj += ':{}'.format(format_spec)
                    result.append('{' + obj + '}')
                else:
                    used_args.add(arg_used)

                    # do any conversion on the resulting object
                    obj = self.convert_field(obj, conversion)

                    # expand the format spec, if needed
                    format_spec, auto_arg_index = self._vformat(
                        format_spec, args, kwargs,
                        used_args, recursion_depth - 1,
                        auto_arg_index=auto_arg_index)

                    # format the object and append to the result
                    result.append(self.format_field(obj, format_spec))

        return ''.join(result), auto_arg_index


def test_partial_format():
    import pytest

    def test_auto_indexing():
        # test basic arg auto-indexing
        assert partial_format('{}{}', 4, 2) == '42'
        assert partial_format('{}{} {}', 4, 2) == '42 {}'

    def test_manual_indexing():
        # test basic arg indexing
        assert partial_format('{0}{1} is not {1} or {0}', 4, 2) == '42 is not 2 or 4'
        assert partial_format('{0}{1} is {3} {1} or {0}', 4, 2) == '42 is {3} 2 or 4'

    def test_mixing_manualauto_fails():
        # test mixing manual and auto args raises
        with pytest.raises(ValueError):
            assert partial_format('{!r} is {0}{1}', 4, 2)

    def test_kwargs():
        # test basic kwarg
        assert partial_format('{base}{n}', base=4, n=2) == '42'
        assert partial_format('{base}{n}', base=4, n=2, extra='foo') == '42'
        assert partial_format('{base}{n} {key}', base=4, n=2) == '42 {key}'

    def test_args_and_kwargs():
        # test mixing args/kwargs with leftovers
        assert partial_format('{}{k} {v}', 4, k=2) == '42 {v}'

        # test mixing with leftovers
        r = partial_format('{}{} is the {k} to {!r}', 4, 2, k='answer')
        assert r == '42 is the answer to {!r}'

    def test_coercion():
        # test coercion is preserved for skipped elements
        assert partial_format('{!r} {k!r}', '42') == "'42' {k!r}"

    def test_nesting():
        # test nesting works with or with out parent keys
        assert partial_format('{k:>{size}}', k=42, size=3) == ' 42'
        assert partial_format('{k:>{size}}', size=3) == '{k:>3}'

    test_mixing_manualauto_fails()
    test_auto_indexing()
    test_manual_indexing()
    test_kwargs()
    test_args_and_kwargs()
    test_coercion()
    test_nesting()

    cases = [
        ('{a} {b}', '1 2.0'),
        ('{z} {y}', '{z} {y}'),
        ('{a} {a:2d} {a:04d} {y:2d} {z:04d}', '1  1 0001 {y:2d} {z:04d}'),
        ('{a!s} {z!s} {d!r}', '1 {z!s} {\'k\': \'v\'}'),
        ('{a!s:>2s} {z!s:>2s}', ' 1 {z!s:>2s}'),
        ('{a!s:>{a}s} {z!s:>{z}s}', '1 {z!s:>{z}s}'),
        ('{a.imag} {z.y}', '0 {z.y}'),
        ('{e[0]:03d} {z[0]:03d}', '042 {z[0]:03d}'),
    ]

    for s, expected in cases:
        # test a bunch of random stuff
        data = dict(
            a=1,
            b=2.0,
            c='3',
            d={'k': 'v'},
            e=[42],
        )
        result = partial_format(s, **data)
        assert expected == result
