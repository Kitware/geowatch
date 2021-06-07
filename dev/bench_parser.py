def benchmark_parse():
    import parse
    import ubelt as ub
    pattern = '{MMM}_{MSIXXX}_{YYYYMMDDHHMMSS}_{Nxxyy}_{ROOO}_{Txxxxx}_{Discriminator}'

    @ub.memoize
    def parser_lut(pattern):
        parser = parse.Parser(pattern)
        return parser

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('memoized parse.Parser'):
        with timer:
            parser = parser_lut(pattern)
            parser.parse('foo')

    for timer in ti.reset('parse.parse'):
        with timer:
            parse.parse(pattern, 'foo')

    print('ti.rankings = {}'.format(ub.repr2(
        ti.rankings, nl=1, precision=8, align=':')))

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/bench_parser.py
    """
    benchmark_parse()
