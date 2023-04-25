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
        >>> from check_docs_references import *  # NOQA
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
    rel_paths = []
    print('repo_dpath = {}'.format(ub.urepr(repo_dpath, nl=1)))
    for r, ds, fs in doc_dpath.walk():
        rel_root = r.relative_to(repo_dpath)
        for f in fs:
            fpath = r / f
            if fpath.suffix == '.rst':
                rel_fpath = rel_root / f
                doc_paths.append((fpath, rel_fpath))
                rel_paths.append(rel_fpath)
            if fpath.suffix == '.md':
                rel_fpath = rel_root / f
                rel_paths.append(rel_fpath)

    doc_paths.append((repo_dpath / 'README.rst', 'README.rst'))
    print('doc_paths = {}'.format(ub.urepr(doc_paths, nl=1)))

    all_links = []
    for fpath, rel_fpath in doc_paths:
        text = fpath.read_text()
        for match in pat.finditer(text):
            group = dict(match.groupdict())
            group['fpath'] = fpath
            all_links.append(group)

    name_to_rel_paths = ub.group_items(rel_paths, lambda p: p.name)

    broken = []
    working = []
    for group in all_links:
        if group['link'].startswith('https'):
            continue

        pointer = group['fpath'].parent / group['link']
        if pointer.exists():
            working.append(group)
        else:
            broken.append(group)

            broken_name = ub.Path(group['link']).name
            if broken_name in name_to_rel_paths:
                candidates = name_to_rel_paths[broken_name]
                if len(candidates) == 1:
                    group['suggestion'] = candidates[0]
                else:
                    group['candidates'] = candidates

    for fpath, subgroups in ub.group_items(broken, key=lambda g: g['fpath']).items():
        print('-----')
        print('fpath = {}'.format(ub.urepr(fpath, nl=1)))
        for group in subgroups:
            print('broken group = {}'.format(ub.urepr(group, nl=1)))

    for fpath, subgroups in ub.group_items(broken, key=lambda g: g['fpath']).items():
        orig_text = fpath.read_text()
        new_text = orig_text
        import os
        prefix = ub.Path(os.path.relpath(repo_dpath, fpath.parent))
        # prefix = repo_dpath.relative_to(fpath.parent)
        for group in subgroups:
            if 'suggestion' in group:
                new_text = new_text.replace(str(group['link']), str(prefix / group['suggestion']))
        if orig_text != new_text:
            print('-----')
            print('fpath = {}'.format(ub.urepr(fpath, nl=1)))
            import xdev
            print(xdev.difftext(orig_text, new_text, colored=True))
            from rich.prompt import Confirm
            if Confirm.ask('Accept this change?'):
                fpath.write_text(new_text)

    print(f'Found {len(working)} working links')
    print(f'Found {len(broken)} broken links')



if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/maintain/check_docs_references.py
    """
    main()
