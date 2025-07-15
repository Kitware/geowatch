#!/usr/bin/env python3
"""
SeeAlso:
    git log --all --format='%aN <%aE>' | sort -u
"""
import scriptconfig as scfg
import ubelt as ub


class GenerateAuthorsConfig(scfg.DataConfig):
    repo_root = scfg.Value('.', help='root of the repo')


KNOWN_ENTITIES = [
    {'email': 'aan244@csr.uky.edu', 'keyname': 'Aram Ansary Ogholbake'},
    {'email': 'ahadzic@dzynetech.com', 'keyname': 'Armin Hadzic'},
    {'email': 'aupadhyaya@dzynetech.com', 'keyname': 'Ajay Upadhyaya'},
    {'email': 'bane.sullivan@kitware.com', 'keyname': 'Bane Sullivan'},
    {'email': 'benjaminbrodie21@gmail.com', 'keyname': 'Benjamin Brodie'},
    {'email': 'cgarchbold@gmail.com', 'keyname': 'Cohen Archbold'},
    {'email': 'david.joy@kitware.com', 'keyname': 'David Joy'},
    {'email': 'dlau@dzynetech.com', 'keyname': 'Dexter Lau'},
    {'email': 'jacob.derosa@kitware.com', 'keyname': 'Jacob DeRosa'},
    {'email': 'jacobbirge24@gmail.com', 'keyname': 'Jacob Birge'},
    {'email': 'ji.suh@uconn.edu', 'keyname': 'Ji Won Suh'},
    {'email': 'matthew.bernstein@kitware.com', 'keyname': 'Matthew Bernstein'},

    {'email': 'atrias@dzynetech.com', 'keyname': 'Antonio Trias'},
    {'email': 'atrias@local.dzynetech.com', 'keyname': 'Antonio Trias'},

    {'email': 'wpines@clarityinnovates.com', 'keyname': 'Wade Pines'},

    {'email': 'matthew.purri@rutgers.edu', 'keyname': 'Matthew Purri'},

    {'email': 'matthew.purri@rutgers.edu', 'keyname': 'Matthew Purri'},
    {'email': 'mjp372@scarletmail.rutgers.edu', 'keyname': 'Matthew Purri'},

    {'email': 'peri.akiva@rutgers.edu', 'keyname': 'Peri Akiva'},
    {'email': 'periha@gmail.com', 'keyname': 'Peri Akiva'},

    {'email': 'ryanlaclair@gmail.com', 'keyname': 'Ryan LaClair'},
    {'email': 's.sastry@wustl.edu', 'keyname': 'Srikumar Sastry'},
    {'email': 'skakun@umd.edu', 'keyname': 'Sergii Skakun'},
    {'email': 'sworkman@dzynetech.com', 'keyname': 'Scott Workman'},

    {'email': 'usman.rafique@kitware.com', 'keyname': 'Usman Rafique'},
    {'email': 'usman.rafique@horologic.khq.kitware.com', 'keyname': 'Usman Rafique'},

    {'email': 'connor.greenwell@kitware.com', 'keyname': 'Connor Greenwell'},
    {'email': 'connor.greenwell@horologic.khq.kitware.com', 'keyname': 'Connor Greenwell'},
    {'email': 'connor.greenwell@arthur.khq.kitware.com', 'keyname': 'Connor Greenwell'},
    {'email': 'connor.greenwell@kwintern-1811.khq.kitware.com', 'keyname': 'Connor Greenwell'},
    {'email': 'cgree3@gmail.com', 'keyname': 'Connor Greenwell'},

    {'email': 'jon.crall@kitware.com' , 'keyname': 'Jon Crall'},
    {'email': 'erotemic@gmail.com', 'keyname': 'Jon Crall'},

    {'email': 'dennis.melamed@kitware.com', 'keyname': 'Dennis Melamed'},
    {'email': 'ben.boeckel@kitware.com', 'keyname': 'Ben Boeckel'},
    {'email': 'christopher.funk@kitware.com', 'keyname': 'Christopher Funk'},
    {'email': 'paul.tunison@kitware.com', 'keyname': 'Paul Tunison'},
    {'email': 'alex.mankowski@kitware.com', 'keyname': 'Alex Mankowski'},

    {'email': 'crallj@rpi.edu', 'keyname': 'Jon Crall'},
    {'email': '49699333+dependabot[bot]@users.noreply.github.com', 'keyname': 'dependabot[bot]'},
    {'email': 'mgorny@gentoo.org', 'keyname': 'Michał Górny'},
    {'email': 'erezshin@gmail.com', 'keyname': 'Erez Shinan'},

    {'email': 'vincenzo.dimatteo@khq-1881.khq.kitware.com', 'keyname': 'Vinnie DiMatteo'},
    {'email': 'vincenzo.dimatteo@horologic.khq.kitware.com', 'keyname': 'Vinnie DiMatteo'},

    {'email': 'ci@circleci.com', 'keyname': 'CircleCI'},
    {'email': 'gaphor@gmail.com', 'keyname': 'Arjan Molenaar'},
    {'email': 'dirk@dmllr.de', 'keyname': 'Dirk Mueller'},
    {'email': 'edgarrm358@gmail.com', 'keyname': 'Edgar Ramírez Mondragón'},
    {'email': 'jayvdb@gmail.com', 'keyname': 'John Vandenberg'},
    {'email': 'mats.lan.pod@googlemail.com', 'keyname': 'Matthias Lambrecht'},
]


def generate_mailmap():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch/dev/maintain'))
    from generate_authors import *  # NOQA
    """

    existing_name_email_pairs = []
    for line in ub.cmd('git shortlog --summary --numbered --email').stdout.split('\n'):
        if line:
            num, pair = line.split('\t')
            name, b = pair.split(' <')
            email = b.strip()[:-1]
            pair = (name, email)
            existing_name_email_pairs.append(pair)

    keyname_to_group = ub.group_items(KNOWN_ENTITIES, key=lambda r: r['keyname'])
    email_to_keyname = {r['email']: r['keyname'] for r in KNOWN_ENTITIES}

    mailmap_lines = []
    unknown_candidates = []
    for name, email in existing_name_email_pairs:
        if email in email_to_keyname:
            keyname = email_to_keyname[email]
            group = keyname_to_group[keyname]
            if group:
                cannon = group[0]
                part1 = f"{cannon['keyname']} <{cannon['email']}>"
                part2 = f"{name} <{email}>"
                if part1 != part2:
                    new_line = f'{part1} {part2}'
                    mailmap_lines.append(new_line)
        else:
            candidate = {'email': email, 'keyname': name}
            unknown_candidates.append(candidate)
    print('unknown_candidates = {}'.format(ub.urepr(unknown_candidates, nl=1)))

    text = '\n'.join(sorted(mailmap_lines)) + '\n'
    mailmap_fpath = ub.Path('.mailmap')
    mailmap_fpath.write_text(text)

    if 0:
        ub.cmd('git add .mailmap', check=1)
        ub.cmd('git commit -m "Add mailmap"', check=1)


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    import pandas as pd
    import parse
    from rich import print
    config = GenerateAuthorsConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = ' + ub.urepr(dict(config), nl=1))

    info = ub.cmd('git shortlog -e --summary --numbered', cwd=config.repo_root)
    lines = [x.strip() for x in info['out'].split('\n') if x.strip()]
    parser = parse.Parser('{num:d}\t{name} <{email}>')

    rows = []
    for line in lines:
        result = parser.parse(line)
        row = dict(result.named)
        print(f'row = {ub.urepr(row, nl=1)}')
        row['id'] = row['keyname'].lower().replace(' ', '')
        rows.append(row)

    df = pd.DataFrame(rows)

    grouped_rows = []
    for _, group in df.groupby('email'):
        new_row = {
        }
        new_row['num'] = group['num'].sum()
        new_row['id'] = group['id'].iloc[0]
        new_row['keyname'] = max(group['keyname'].tolist(), key=len)
        new_row['email'] = group['email'].iloc[0]
        grouped_rows.append(new_row)

    df = pd.DataFrame(grouped_rows)

    grouped_rows = []
    for _, group in df.groupby('id'):
        new_row = {
        }
        new_row['num'] = group['num'].sum()
        new_row['id'] = group['id'].iloc[0]
        new_row['keyname'] = max(group['keyname'].tolist(), key=len)
        new_row['email'] = group['email'].iloc[0]
        grouped_rows.append(new_row)

    df = pd.DataFrame(grouped_rows)
    df = df.sort_values('num', ascending=False)
    print(df)
    print(', '.join(df['keyname']))
    # print(ub.urepr(df.drop(['id', 'num'], axis=1).to_dict('records')))


def author_stats(repo):
    log_info = repo.cmd("git log --format='author: %ae' --numstat")
    # log_info = repo.cmd("git log --since='1 year ago' --format='author: %ae' --numstat")
    print(log_info.stdout)
    author_stats = ub.ddict(lambda: ub.ddict(int))
    author_files = ub.ddict(set)
    author = None
    for line in log_info.stdout.split('\n'):
        line_ = line.strip()
        if line_:
            if line.startswith('author: '):
                author = line.split(' ')[1]
                author_stats[author]['commits'] += 1
            else:
                inserts, deletes, fpath = line.split('\t')
                inserts = int(0 if inserts == '-' else inserts)
                deletes = int(0 if deletes == '-' else deletes)
                total = inserts + deletes
                author_stats[author]['inserts'] += inserts
                author_stats[author]['deletes'] += deletes
                author_stats[author]['total'] += total
                author_files[author].add(fpath)

    author_stats = ub.udict(author_stats).sorted_values(lambda v: v['commits'])

    author_alias = {}
    for r in KNOWN_ENTITIES:
        author_alias[r['email']] = r['keyname']
        author_alias[r['keyname']] = r['keyname']

    rows = []
    for author, stats in author_stats.items():
        name = author_alias.get(author, author)
        row = {'author': author}
        row['keyname'] = name
        row.update(stats)
        rows.append(row)

    import pandas as pd
    import rich
    df = pd.DataFrame(rows)

    final_df = df.groupby('keyname').sum()
    final_df = final_df.sort_values('commits')
    rich.print(final_df)

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/dev/maintain/generate_authors.py
        python -m generate_authors
    """
    main()
