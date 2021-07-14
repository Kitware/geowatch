
def custom_lint(dpath: str):
    """
    Runs our custom "watch" linting rules on a specific directory.

    Args:
        dpath (str): the path to lint
    """
    import ubelt as ub
    flake8_errors = [
        'E123',  # closing braket indentation
        'E126',  # continuation line hanging-indent
        'E127',  # continuation line over-indented for visual indent
        'E201',  # whitespace after '('
        'E202',  # whitespace before ']'
        'E203',  # whitespace before ', '
        'E221',  # multiple spaces before operator  (TODO: I wish I could make an exception for the equals operator. Is there a way to do this?)
        'E222',  # multiple spaces after operator
        'E241',  # multiple spaces after ,
        'E265',  # block comment should start with "# "
        'E271',  # multiple spaces after keyword
        'E272',  # multiple spaces before keyword
        'E301',  # expected 1 blank line, found 0
        'E305',  # expected 1 blank line after class / func
        'E306',  # expected 1 blank line before func
        #'E402',  # module import not at top
        'E501',  # line length > 79
        'W602',  # Old reraise syntax
        'E266',  # too many leading '#' for block comment
        'N801',  # function name should be lowercase [N806]
        'N802',  # function name should be lowercase [N806]
        'N803',  # argument should be lowercase [N806]
        'N805',  # first argument of a method should be named 'self'
        'N806',  # variable in function should be lowercase [N806]
        'N811',  # constant name imported as non constant
        'N813',  # camel case
        'W503',  # line break before binary operator
        'W504',  # line break after binary operator

        'I201',  # Newline between Third party import groups
        'I100',  # Wrong import order
    ]
    flake8_args_list = [
        '--max-line-length', '79',
        '--ignore=' + ','.join(flake8_errors)
    ]
    flake8_exe = 'flake8'
    info = ub.cmd([flake8_exe] + flake8_args_list + [dpath], verbose=1)
    return info['ret']


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/lint.py --help

        cd $HOME/code/watch
        python ~/code/watch/dev/lint.py watch
    """
    import fire
    fire.Fire(custom_lint)


def demo_stac_catalog():
    import json
    import ubelt as ub
    import os
    dpath = ub.ensure_app_cache_dir('watch/demo/demo_stac')
    fpath = os.path.join(dpath, 'demo_catalog.json')
    expected_sha512 = '1b813bfe5661963c08dff322734f8d34a6d8b06bcc15fc4'

    if os.path.exists(fpath):
        # Dont rewrite if we dont need to
        got_sha512 = ub.hash_file(fpath)
        if got_sha512.startswith(expected_sha512):
            return fpath

    catalog = {
        'some': {'json': 'structure'},
    }
    with open(fpath, 'w') as file:
        json.dump(catalog, file)
    return fpath
