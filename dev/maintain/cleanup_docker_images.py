#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CleanupDockerImagesCLI(scfg.DataConfig):
    """
    For images that match the reposity pattern, remove all but the latest image
    for every major version.

    Before running this, be sure to remove containers you may want to delete
    the images for.  To remove all containers you can use the following bash
    command.

    .. code:: bash
        docker stop $(docker ps -a -q)
        docker rm $(docker ps -a -q)
    """
    repo_pattern = scfg.Value(
        '*watch', help=(
            'a pattern that specifies the repo of interest. '
            'This defaults to something useful for the SMART project'))

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/watch/dev/maintain'))
            >>> from cleanup_docker_images import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CleanupDockerImagesCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        from kwutil import util_pattern
        repo_pat = util_pattern.Pattern.coerce(config.repo_pattern)

        info = ub.cmd('docker images --format=json')
        info.check_returncode()
        import json
        rows = []
        for line in info['out'].split('\n'):
            if line.strip():
                row = json.loads(line)
                rows.append(row)

        candidates = []
        for row in rows:
            if repo_pat.match(row['Repository']):
                candidates.append(row)

        groups = ub.group_items(candidates, key=lambda x: (x['Repository'], x['Tag'].split('-')[0]))
        to_remove = []
        to_keep = []
        print(ub.urepr(ub.udict(groups).map_values(len)))
        for key, group in groups.items():
            group = sorted(group, key=lambda x: x['CreatedAt'])
            to_remove.extend(group[:-1])
            to_keep.extend(group[-1])
        print(f'{len(to_keep)=}')
        print(f'{len(to_remove)=}')

        import rich.prompt
        ans = rich.prompt.Confirm.ask('Remove these images?')
        if ans:
            remove_ids = [r['Repository'] + ':' + r['Tag'] for r in to_remove]
            ub.cmd(['docker', 'image', 'remove'] + remove_ids, verbose=3)

        if 0:
            import pandas as pd
            df = pd.DataFrame(rows)
            rich.print(df.to_string())

__cli__ = CleanupDockerImagesCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/maintain/cleanup_docker_images.py
        python -m cleanup_docker_images
    """
    main()
