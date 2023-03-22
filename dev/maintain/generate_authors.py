#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class GenerateAuthorsConfig(scfg.DataConfig):
    repo_root = scfg.Value('.', help='root of the repo')


KNOWN_ENTITIES = [
    {'email': 'aan244@csr.uky.edu', 'name': 'Aram Ansary Ogholbake'},
    {'email': 'ahadzic@dzynetech.com', 'name': 'Armin Hadzic'},
    {'email': 'aupadhyaya@dzynetech.com', 'name': 'Ajay Upadhyaya'},
    {'email': 'bane.sullivan@kitware.com', 'name': 'Bane Sullivan'},
    {'email': 'benjaminbrodie21@gmail.com', 'name': 'Benjamin Brodie'},
    {'email': 'cgarchbold@gmail.com', 'name': 'Cohen Archbold'},
    {'email': 'connor.greenwell@kitware.com', 'name': 'Connor Greenwell'},
    {'email': 'david.joy@kitware.com', 'name': 'David Joy'},
    {'email': 'dlau@dzynetech.com', 'name': 'Dexter Lau'},
    {'email': 'jacob.derosa@kitware.com', 'name': 'Jacob DeRosa'},
    {'email': 'jacobbirge24@gmail.com', 'name': 'Jacob Birge'},
    {'email': 'ji.suh@uconn.edu', 'name': 'Ji Won Suh'},
    {'email': 'jon.crall@kitware.com' , 'name': 'Jon Crall'},
    {'email': 'matthew.bernstein@kitware.com', 'name': 'Matthew Bernstein'},
    {'email': 'matthew.purri@rutgers.edu', 'name': 'Matthew Purri'},
    {'email': 'peri.akiva@rutgers.edu', 'name': 'Peri Akiva'},
    {'email': 'ryanlaclair@gmail.com', 'name': 'Ryan LaClair'},
    {'email': 's.sastry@wustl.edu', 'name': 'Srikumar Sastry'},
    {'email': 'skakun@umd.edu', 'name': 'Sergii Skakun'},
    {'email': 'sworkman@dzynetech.com', 'name': 'Scott Workman'},
    {'email': 'usman.rafique@kitware.com', 'name': 'Usman Rafique'},
]


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
        row['id'] = row['name'].lower().replace(' ', '')
        rows.append(row)

    df = pd.DataFrame(rows)

    grouped_rows = []
    for _, group in df.groupby('email'):
        new_row = {
        }
        new_row['num'] = group['num'].sum()
        new_row['id'] = group['id'].iloc[0]
        new_row['name'] = max(group['name'].tolist(), key=len)
        new_row['email'] = group['email'].iloc[0]
        grouped_rows.append(new_row)

    df = pd.DataFrame(grouped_rows)

    grouped_rows = []
    for _, group in df.groupby('id'):
        new_row = {
        }
        new_row['num'] = group['num'].sum()
        new_row['id'] = group['id'].iloc[0]
        new_row['name'] = max(group['name'].tolist(), key=len)
        new_row['email'] = group['email'].iloc[0]
        grouped_rows.append(new_row)

    df = pd.DataFrame(grouped_rows)
    df = df.sort_values('num', ascending=False)
    print(df)
    print(', '.join(df['name']))
    # print(ub.repr2(df.drop(['id', 'num'], axis=1).to_dict('records')))

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/maintain/generate_authors.py
        python -m generate_authors
    """
    main()
