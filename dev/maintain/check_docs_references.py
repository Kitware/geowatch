#!/usr/bin/env python3
"""
Check that the references linked in the docs are all consistent
"""
import scriptconfig as scfg
import ubelt as ub


class CheckDocsConfig(scfg.DataConfig):
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
    import rich
    import re
    import os
    config = CheckDocsConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    repo_dpath = ub.Path(config.repo_dpath)
    rich.print('config = ' + ub.urepr(dict(config), nl=1))

    if 0:
        import xdev
        b = xdev.regex_builder.RegexBuilder.coerce('python')
        b.escape('`')
        text_pat = b.named_field('[^<>`]' + b.nongreedy, 'rst_text')
        link_pat = b.named_field('[^<>`]' + b.nongreedy, 'rst_link')
        no_doubletick_before = b.lookbehind('`', positive=False)
        no_doubletick_after = b.lookahead('`', positive=False)
        pattern = f'{no_doubletick_before}`{text_pat}<{link_pat}>`{no_doubletick_after}'
        print('pattern = {}'.format(ub.urepr(pattern, nl=1)))

    # Define a pattern to match RST links
    rst_link_pat = re.compile(
        '(?<!`)`(?P<rst_text>[^<>`]*?)<(?P<rst_link>[^<>`]*?)>`(?!`)'
    )

    # Enumerate the paths to all files in the docs folder.
    # Or more generally, all the files we are interested in checking or linking
    # to.
    doc_paths = []
    repo_dpath = repo_dpath.absolute()
    doc_dpath = repo_dpath / 'docs'
    rel_paths = []
    abs_paths = []
    print('repo_dpath = {}'.format(ub.urepr(repo_dpath, nl=1)))
    for r, ds, fs in doc_dpath.walk():
        rel_root = r.relative_to(doc_dpath)
        for f in fs:
            fpath = r / f
            if fpath.suffix == '.rst':
                rel_fpath = rel_root / f
                doc_paths.append((fpath, rel_fpath))
                rel_paths.append(rel_fpath)
                abs_paths.append(fpath)
            if fpath.suffix == '.md':
                rel_fpath = rel_root / f
                rel_paths.append(rel_fpath)
                abs_paths.append(fpath)

    doc_paths.append((repo_dpath / 'README.rst', 'README.rst'))
    print('doc_paths = {}'.format(ub.urepr(doc_paths, nl=1)))

    # Read all of the documentation files and search for links in the text
    all_links = []
    for fpath, rel_fpath in doc_paths:
        text = fpath.read_text()
        for match in rst_link_pat.finditer(text):
            group = dict(match.groupdict())
            group['fpath'] = fpath
            all_links.append(group)

    # name_to_rel_paths = ub.group_items(rel_paths, lambda p: p.name)
    name_to_abs_paths = ub.group_items(abs_paths, lambda p: p.name)
    print('name_to_abs_paths = {}'.format(ub.urepr(name_to_abs_paths, nl=1)))

    # For all found links, determine if they exist or if they are broken.
    # For broken links, we can try to find suggested fixes.
    broken = []
    working = []
    for group in all_links:
        if group['rst_link'].startswith('https'):
            continue

        rst_link = ub.Path(group['rst_link'])
        parent_dpath = group['fpath'].parent
        pointer = parent_dpath  / rst_link
        if pointer.exists():
            working.append(group)
        else:
            broken.append(group)
            # Use heuristics to see if we can fix the link.
            broken_name = rst_link.name
            suggestion = None
            candidates = []
            if broken_name in name_to_abs_paths:
                candidates.extend(name_to_abs_paths[broken_name])

            # if rel_link.parts[0:1] == ['docs']:
            #     cand = parent_dpath / ub.Path(rel_link.parts[1:])
            #     if cand.exists():
            #         candidates.append(cand)

            if len(candidates) == 1:
                suggestion = candidates[0]
            if suggestion is not None:
                group['suggestion'] = suggestion
                import xdev
                with xdev.embed_on_exception_context:
                    # os.path.relpath(parent_dpath, suggestion)
                    # suggestion.resolve().relative_to(parent_dpath.resolve())
                    group['rel_suggestion'] = os.path.relpath(suggestion, parent_dpath)
            elif candidates:
                group['candidates'] = candidates

    # Print a report of what we found
    for fpath, subgroups in ub.group_items(broken, key=lambda g: g['fpath']).items():
        print('-----')
        print('fpath = {}'.format(ub.urepr(fpath, nl=1)))
        for group in subgroups:
            print('broken group = {}'.format(ub.urepr(group, nl=1)))

    # Attempt to fix links where suggestions were made.
    # docs_dpath = repo_dpath / 'docs'

    for fpath, subgroups in ub.group_items(broken, key=lambda g: g['fpath']).items():
        orig_text = fpath.read_text()
        new_text = orig_text
        parent_dpath = fpath.parent
        # prefix = ub.Path(os.path.relpath(docs_dpath, fpath.parent))
        # prefix = repo_dpath.relative_to(fpath.parent)
        for group in subgroups:
            if 'suggestion' in group:
                old_link = str(group['rst_link'])
                new_link = str(os.path.relpath(group['suggestion'], parent_dpath))
                print(f'old_link={old_link}')
                print(f'new_link={new_link}')
                # new_link = str(os.path.normpath(prefix / group['suggestion']))
                new_text = new_text.replace(old_link, new_link)
        if orig_text != new_text:
            print('-----')
            print(f'parent_dpath={parent_dpath}')
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
