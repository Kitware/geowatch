#!/usr/bin/env python3
"""
Check that the references linked in the docs are all consistent
"""
import scriptconfig as scfg
import ubelt as ub


class MyNewConfig(scfg.DataConfig):
    repo_dpath = scfg.Value('.', help='input')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch/dev/maintain'))
        >>> from check_doc_references import *  # NOQA
        >>> import watch
        >>> repo_dpath = ub.Path(watch.__file__).parent.parent
        >>> cmdline = 0
        >>> kwargs = dict(repo_dpath=repo_dpath)
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = MyNewConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    repo_dpath = ub.Path(config.repo_dpath)
    print('config = ' + ub.urepr(dict(config), nl=1))

    import re
    if 0:
        import xdev
        b = xdev.regex_builder.RegexBuilder.coerce('python')
        b.escape('`')
        text_pat = b.named_field('[^<>`]' + b.nongreedy, 'text')
        link_pat = b.named_field('[^<>`]' + b.nongreedy, 'link')
        no_doubletick_before = b.lookbehind('`', positive=False)
        no_doubletick_after = b.lookahead('`', positive=False)
        pattern = f'{no_doubletick_before}`{text_pat}<{link_pat}>`{no_doubletick_after}'
        print('pattern = {}'.format(ub.urepr(pattern, nl=1)))
    pattern = '(?<!`)`(?P<text>[^<>`]*?)<(?P<link>[^<>`]*?)>`(?!`)'
    pat = re.compile(pattern)

    doc_paths = []
    repo_dpath = repo_dpath.absolute()
    doc_dpath = repo_dpath / 'docs'
    for r, ds, fs in doc_dpath.walk():
        rel_root = r.relative_to(repo_dpath)
        for f in fs:
            fpath = r / f
            if fpath.suffix == '.rst':
                rel_fpath = rel_root / f
                doc_paths.append((fpath, rel_fpath))
    doc_paths.append((repo_dpath / 'README.rst', 'README.rst'))

    all_links = []
    for fpath, rel_fpath in doc_paths:
        text = fpath.read_text()
        for match in pat.finditer(text):
            group = dict(match.groupdict())
            group['fpath'] = fpath
            all_links.append(group)

    for group in all_links:
        if group['link'].startswith('https'):
            continue

        pointer = group['fpath'].parent / group['link']
        if not pointer.exists():
            print('group = {}'.format(ub.urepr(group, nl=1)))

        ...

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/maintain/check_doc_references.py
        python -m check_doc_references
    """
    main()
