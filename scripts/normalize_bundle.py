"""
Normalize a kwcoco bundle in DVC

WIP
"""


#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class NormalizeBundleCLI(scfg.DataConfig):
    bundle_dpath = scfg.Value(None, help='path to kwcoco bundle')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from normalize_bundle import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict(bundle_dpath='.')
            >>> kwargs = dict(bundle_dpath='/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc/Aligned-Drop7')
            >>> cls = NormalizeBundleCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        import networkx as nx
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))
        bundle_dpath = ub.Path(config.bundle_dpath)

        bundle_dpath = bundle_dpath.resolve()

        import simple_dvc
        dvc = simple_dvc.SimpleDVC.coerce(bundle_dpath)
        dvc_sidecars = list(dvc.sidecar_paths(bundle_dpath))

        tracked_dpaths = {
            _p for p in dvc_sidecars
            if (_p := p.augment(ext='')).is_dir()
        }
        tracked_fpaths = {
            _p for p in dvc_sidecars
            if (_p := p.augment(ext='')).is_file()
        }

        KNOWN_SENSOR_CODES = {'S2', 'L8', 'WV', 'WV1', 'PD'}
        KNOWN_MULTISENSOR_CODES = {'raw_bands'}

        # A list of dictionaries that keeps information about untracked paths
        untracked_rows = []
        still_untracked = nx.DiGraph()
        for r, ds, fs in bundle_dpath.walk():

            # Dont recurse into tracked directories
            if r in tracked_dpaths:
                ds.clear()
                continue

            # Check if this directory should be tracked
            if r.name in KNOWN_SENSOR_CODES:
                if 'affine_warp' in ds:
                    # This is a sensor rawband directory that should be tracked
                    untracked_rows.append({
                        'type': 'rawband-sensor-dir',
                        'path': r,
                    })
                    ds.clear()
                    continue

            if r.name in KNOWN_MULTISENSOR_CODES:
                if any(d.startswith('ave_') for d in ds) or any(r.ls('*/ave_*')):
                    # This is an averaged rawband dir
                    untracked_rows.append({
                        'type': 'rawband-multisensor-averaged-dir',
                        'path': r,
                    })
                    ds.clear()
                    continue

            if r.parent != r:
                still_untracked.add_edge(r.parent, r)

            for f in fs:
                fpath = r / f
                if fpath not in tracked_fpaths:
                    if not f.endswith('.dvc') and not f.startswith('.'):
                        untracked_rows.append({
                            'type': 'kwcoco',
                            'path': fpath,
                        })
        if 0:
            import pandas as pd
            print(pd.DataFrame(untracked_rows)['type'].value_counts())

        nx.write_network_text(still_untracked)

        MODIFY_DVC_STUFF = 0
        needs_dvc_add = []
        type_to_rows = ub.group_items(untracked_rows, key=lambda x: x['type'])

        untracked_fpaths = [r['path'] for r in type_to_rows['kwcoco']]
        needs_dvc_add += [f for f in untracked_fpaths if f.endswith('.kwcoco.zip')]

        for row in type_to_rows.get('rawband-multisensor-averaged-dir', []):
            needs_dvc_add.append(row['path'])

        # for needs_add_dpath in untracked_rawband_dirs:
        #     print(needs_add_dpath.ls())

        if MODIFY_DVC_STUFF:
            dvc.add(needs_dvc_add, verbose=2)

        top_dpaths = [p for p in bundle_dpath.ls() if p.is_dir()]
        region_dpaths = []
        for dpath in top_dpaths:
            # not robust
            if len(dpath.name.split('_')[0]) == 2:
                region_dpaths.append(dpath)

        region_rows = []
        for dpath in region_dpaths:
            children = list(dpath.ls())
            sensor_dirs = []
            coco_fpaths = []
            for p in children:
                if p.name in KNOWN_SENSOR_CODES:
                    sensor_dirs.append(p)
                if p.endswith('.kwcoco.zip'):
                    coco_fpaths.append(p)
            unhandled = set(children) - (set(coco_fpaths) | set(sensor_dirs))
            print('unhandled = {}'.format(ub.urepr(unhandled, nl=1)))
            row = {
                'region_id': dpath.name,
                # 'dpath': dpath,
                'sensor_dirs': [p.name for p in sensor_dirs],
                'coco_fpaths': [p.name for p in coco_fpaths],
            }
            region_rows.append(row)

        if 0:
            import pandas as pd
            print(pd.DataFrame(region_rows))


def _devcheck(tracked_dpaths, bundle_dpath):
    import networkx as nx
    if 0:
        untracked = nx.DiGraph()
        for r, ds, fs in bundle_dpath.walk():
            tracked_idxs = [idx for idx, d in enumerate(ds) if (r / d) in tracked_dpaths]
            for idx in reversed(tracked_idxs):
                del ds[idx]

            for d in ds:
                dpath = r / d
                untracked.add_edge(r, dpath)
        nx.write_network_text(untracked)


__cli__ = NormalizeBundleCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/scripts/normalize_bundle.py
        python -m normalize_bundle
    """
    main()
