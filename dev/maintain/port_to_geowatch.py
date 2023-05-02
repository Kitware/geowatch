# SeeAlso:
# ~/code/watch/dev/maintain/mirror_package_geowatch.py


def main():
    import ubelt as ub
    old_name = 'watch'
    module = ub.import_module_from_name(old_name)

    module_dpath = ub.Path(module.__file__).parent
    repo_dpath = module_dpath.parent
    repo_dpath / 'docs'

    import xdev
    _ = xdev.grep(r'\bSMART WATCH\b', dpath=module_dpath, include='*.py')
    xdev.sed(r'\bSMART WATCH\b', repl='GEOWATCH', dpath=module_dpath, include='*.py', dry=0)

    import xdev
    b = xdev.regex_builder.RegexBuilder.coerce('python')
    no_geo = b.lookbehind('GEO-', positive=False)
    xdev.sed(no_geo + r'\bWATCH\b', repl='GEOWATCH', dpath=module_dpath, include='*.py', dry=0)


    docs_dpath = repo_dpath / 'docs'
    xdev.sed(no_geo + r'\bWATCH\b', repl='GEOWATCH', dpath=docs_dpath, include='*.rst', dry=0)
    _ = xdev.grep(r'SMART.*GEOWATCH\b', dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(r'\bwatch\b', dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(r'\bsmartwatch\b', dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(r'\bwatch\b', dpath=docs_dpath, include='*.rst')
    import re
    _ = xdev.grep(re.compile(r'\bwatch\b', flags=re.IGNORECASE), dpath=docs_dpath, include='*.rst')
    import re
    _ = xdev.grep(re.compile(r'\bWatch\b'), dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(re.compile(r'\bWatch\b'), dpath=repo_dpath, include='*.rst')

    _ = xdev.grep(re.compile(r'\bWATCH\b'), dpath=repo_dpath, include='*', exclude=['.git'])
