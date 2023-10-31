# SeeAlso:
# ~/code/watch/dev/maintain/mirror_package_geowatch.py


def replace_watch_with_geowatch_in_module_and_docs_v1():
    """
    We already did this when geowatch was the mirror.
    """
    import ubelt as ub
    import re
    import xdev
    old_name = 'watch'
    module = ub.import_module_from_name(old_name)

    module_dpath = ub.Path(module.__file__).parent
    repo_dpath = module_dpath.parent
    repo_dpath / 'docs'

    _ = xdev.grep(r'\bSMART WATCH\b', dpath=module_dpath, include='*.py')
    xdev.sed(r'\bSMART WATCH\b', repl='GEOWATCH', dpath=module_dpath, include='*.py', dry=0)

    b = xdev.regex_builder.RegexBuilder.coerce('python')
    no_geo = b.lookbehind('GEO-', positive=False)
    xdev.sed(no_geo + r'\bWATCH\b', repl='GEOWATCH', dpath=module_dpath, include='*.py', dry=0)

    docs_dpath = repo_dpath / 'docs'
    xdev.sed(no_geo + r'\bWATCH\b', repl='GEOWATCH', dpath=docs_dpath, include='*.rst', dry=0)
    _ = xdev.grep(r'SMART.*GEOWATCH\b', dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(r'\bwatch\b', dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(r'\bsmartwatch\b', dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(r'\bwatch\b', dpath=docs_dpath, include='*.rst')
    _ = xdev.grep(re.compile(r'\bwatch\b', flags=re.IGNORECASE), dpath=docs_dpath, include='*.rst')
    _ = xdev.grep(re.compile(r'\bWatch\b'), dpath=docs_dpath, include='*.rst')

    _ = xdev.grep(re.compile(r'\bWatch\b'), dpath=repo_dpath, include='*.rst')

    _ = xdev.grep(re.compile(r'\bWATCH\b'), dpath=repo_dpath, include='*', exclude=['.git'])


def replace_watch_with_geowatch_in_module_and_docs_v2():
    """
    This is the real deal that changes imports and primary invocations.

    Prereq:

        # This is a batch script that modifies the 0.10.0 branch
        # to make geowatch the main package and watch the mirror

        mkdir -p $HOME/temp/port

        # Work on a non-installed copy of the repo
        cd $HOME/temp/port
        git clone $HOME/code/watch/.git ./watch

        cd $HOME/temp/port/watch

        git remote add gitlab git@gitlab.kitware.com:smart/watch.git
        git co -b dev/make_geowatch_primary

        cd $HOME/temp/port/watch
        git fetch
        git reset --hard origin/main

    """
    import ubelt as ub
    import xdev

    module_dpath = ub.Path('/home/joncrall/temp/port/watch/geowatch')
    repo_dpath = module_dpath.parent

    # Reset to last working state
    ub.cmd('git reset --hard origin/main', cwd=repo_dpath, verbose=3)

    # Delete the old geowatch mirror
    (module_dpath).delete()

    _ = ub.cmd('git commit -am "delete old geowatch mirror"', cwd=repo_dpath, verbose=3)

    ub.cmd('git mv watch geowatch', cwd=repo_dpath, verbose=3)

    _ = ub.cmd('git commit -am "initial move of watch -> geowatch"', cwd=repo_dpath, verbose=3)

    # Spot check that common patterns dont contain false positives
    if 0:
        _ = xdev.grep(r'from \bwatch\b', dpath=module_dpath)
        _ = xdev.grep(r'import \bwatch\b', dpath=module_dpath)
        _ = xdev.grep(r'-m \bwatch\b', dpath=module_dpath)
        _ = xdev.grep(r'\bwatch\.find_dvc_dpath', dpath=module_dpath)
        _ = xdev.grep(r'\bwatch\.', dpath=module_dpath)
        _ = xdev.grep(r'\'\bwatch\'', dpath=module_dpath)
        _ = xdev.grep(r'\'\bwatch-msi\'', dpath=module_dpath)

        _ = xdev.grep(r'\bwatch/tasks', dpath=module_dpath)
        _ = xdev.grep(r'\bwatch/gis', dpath=module_dpath)
        _ = xdev.grep(r'\bwatch/cli', dpath=module_dpath)
        _ = xdev.grep(r'\bwatch/tests', dpath=module_dpath)

    # Execute search / replace

    def main_search_replace(dpath):
        xdev.sed(r'from \bwatch\b', repl='from geowatch', dpath=dpath)
        xdev.sed(r'import \bwatch\b', repl='import geowatch', dpath=dpath)
        xdev.sed(r'-m \bwatch\b', repl='-m geowatch', dpath=dpath)
        xdev.sed(r'\bwatch\.find_dvc_dpath', repl=r'geowatch.find_dvc_dpath', dpath=dpath)
        xdev.sed(r'\bwatch\.', repl=r'geowatch.', dpath=dpath)
        xdev.sed(r"'\bwatch'", repl=r"'geowatch'", dpath=dpath)
        xdev.sed(r"'watch-msi'", repl=r"'geowatch-msi'", dpath=dpath)
        xdev.sed(r'\bwatch/tasks', repl=r'geowatch/tasks', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/gis', repl=r'geowatch/gis', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/mlops', repl=r'geowatch/mlops', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/cli', repl=r'geowatch/cli', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/stac', repl=r'geowatch/stac', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/rc', repl=r'geowatch/rc', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/geoannots', repl=r'geowatch/geoannots', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/utils', repl=r'geowatch/utils', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/tests', repl=r'geowatch/tests', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/test', repl=r'geowatch/test', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/demo', repl=r'geowatch/demo', dpath=dpath, dry=0)

        xdev.sed(r'\bwatch/heuristics', repl=r'geowatch/heuristics', dpath=dpath, dry=0)
        xdev.sed(r'\bwatch/hash_rlut', repl=r'geowatch/hash_rlut', dpath=dpath, dry=0)
        xdev.sed(r"\bappname='watch", repl=r"appname='geowatch", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch fields", repl=r"geowatch feilds", dpath=dpath, dry=0)
        xdev.sed(r"\bcache/watch", repl=r"cache/geowatch", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch-msi", repl=r"geowatch-msi", dpath=dpath, dry=0)
        xdev.sed(r"`watch`", repl=r"`geowatch`", dpath=dpath, dry=0)
        xdev.sed(r"vidshapes-watch", repl=r"vidshapes-geowatch", dpath=dpath, dry=0)
        xdev.sed(r"vidshapes8-watch", repl=r"vidshapes8-geowatch", dpath=dpath, dry=0)
        xdev.sed(r"vidshapes2-watch", repl=r"vidshapes2-geowatch", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch-multisensor", repl=r"geowatch-multisensor", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch-teamfeat", repl=r"geowatch-teamfeat", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch util_logging", repl=r"geowatch util_logging", dpath=dpath, dry=0)

        xdev.sed(r"The watch command", repl=r"The geowatch command", dpath=dpath, dry=0)
        xdev.sed(r"the watch heuristics", repl=r"the geowatch heuristics", dpath=dpath, dry=0)
        xdev.sed(r"useful watch stats", repl=r"useful geowatch stats", dpath=dpath, dry=0)
        xdev.sed(r"the watch package", repl=r"the geowatch package", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch module", repl=r"geowatch module", dpath=dpath, dry=0)
        xdev.sed(r"appdir\('watch", repl=r"appdir('geowatch", dpath=dpath, dry=0)

        xdev.sed(r"\bwatch-splits", repl=r"geowatch-splits", dpath=dpath, dry=0)

        xdev.sed(r"docker build -t watch", repl=r"docker build -t geowatch", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch bash", repl=r"geowatch bash", dpath=dpath, dry=0)
        xdev.sed(r"\bwatch-demo-data", repl=r"geowatch-demo-data", dpath=dpath, dry=0)
        xdev.sed(r"XDG_CACHE_HOME/watch", repl=r"XDG_CACHE_HOME/geowatch", dpath=dpath, dry=0)
        xdev.sed(r"conda activate watch", repl=r"conda activate geowatch", dpath=dpath, dry=0)
        xdev.sed(r"for watch lightning", repl=r"for geowatch lightning", dpath=dpath, dry=0)
        xdev.sed(r"to watch to", repl=r"to geowatch to", dpath=dpath, dry=0)
        xdev.sed(r"think watch is", repl=r"think geowatch is", dpath=dpath, dry=0)
        xdev.sed(r"real watch data", repl=r"real geowatch data", dpath=dpath, dry=0)
        xdev.sed(r"the watch DVC", repl=r"the geowatch DVC", dpath=dpath, dry=0)
        xdev.sed(r"installing watch", repl=r"installing geowatch", dpath=dpath, dry=0)
        xdev.sed(r"with watch special", repl=r"with geowatch special", dpath=dpath, dry=0)
        xdev.sed(r"non watch dependant", repl=r"non geowatch dependant", dpath=dpath, dry=0)
        xdev.sed(r"non watch dependant", repl=r"non geowatch dependant", dpath=dpath, dry=0)

    dpath = module_dpath
    main_search_replace(dpath)

    if 0:
        _ = xdev.grep(r'\bwatch\b', dpath=module_dpath)

        # Ignore a false positive in data paths
        b = xdev.regex_builder.RegexBuilder.coerce('python')

        special_excludes = (
            b.lookbehind('/SCRATCH/', positive=False) +
            b.lookbehind('/code/', positive=False) +
            b.lookbehind('/data/', positive=False) +
            b.lookbehind('/projects/', positive=False) +
            b.lookbehind('smart-', positive=False) +
            b.lookbehind('envs/', positive=False) +
            b.lookbehind('lib/', positive=False) +
            b.lookbehind('smart/', positive=False) +
            b.lookbehind('youtube.com/', positive=False) +
            ''
        )
        _ = xdev.grep(special_excludes + r'\bwatch\b', dpath=module_dpath)

    ub.cmd('git commit -am "Search replace watch with geowatch in main module"', cwd=repo_dpath, verbose=3)

    test_dpath = repo_dpath / 'tests'
    dpath = test_dpath
    main_search_replace(dpath)

    ub.cmd('git commit -am "Search replace watch with geowatch in tests"', cwd=repo_dpath, verbose=3)

    import sys, ubelt  # NOQA
    sys.path.append(ubelt.expandpath('~/code/watch/dev/maintain'))
    from mirror_package_geowatch import do_mirror
    module_dpath = ub.Path('/home/joncrall/temp/port/watch/geowatch')
    mirror_name = 'watch'
    do_mirror(module_dpath, mirror_name)

    ub.cmd('git add watch', cwd=repo_dpath, verbose=3)
    ub.cmd('git commit -am "Made mirror for the original watch package"', cwd=repo_dpath, verbose=3)

    setup_fpath = repo_dpath / 'setup.py'
    xdev.sedfile(setup_fpath, r"\bwatch.tasks.depth", repl=r"geowatch.tasks.depth", dry=0)
    xdev.sedfile(setup_fpath, r"\bwatch.cli", repl=r"geowatch.cli", dry=0)
    xdev.sedfile(setup_fpath, r"\bwatch/", repl=r"geowatch/", dry=0)

    fpath = repo_dpath / 'pyproject.toml'
    xdev.sedfile(fpath, r"\bwatch", repl=r"geowatch", dry=0)

    fpath = repo_dpath / 'run_developer_setup.sh'
    xdev.sedfile(fpath, r"\bwatch", repl=r"geowatch", dry=0)

    fpath = repo_dpath / 'run_tests.py'
    xdev.sedfile(fpath, r"\bwatch", repl=r"geowatch", dry=0)

    fpath = repo_dpath / 'run_linter.sh'
    xdev.sedfile(fpath, r"\bwatch", repl=r"geowatch", dry=0)

    ub.cmd('git commit -am "Updated repo files to geowatch"', cwd=repo_dpath, verbose=3)
