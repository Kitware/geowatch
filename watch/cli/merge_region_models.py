"""
"""
import re
import ubelt as ub
import scriptconfig as scfg
import json
import geojson


class MergeRegionModelConfig(scfg.Config):
    """
    Combine the specific features from multiple region files into a single one.
    """
    default = {
        'src': scfg.Value([], help='paths to input geojson region files', nargs='+', position=1),
        'dst': scfg.Value(None, help='path to combined multi-region file'),
        'match_type': scfg.Value('region', help=ub.paragraph(
            '''
            regex that filters results to only contain features where
            properties.type that match.
            ''')),
    }
    epilog = r"""
    Example Usage:

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

        python -m watch merge_region_models \
            --src $DVC_DPATH/drop1/region_models/*.geojson \
            --dst $DVC_DPATH/drop1/all_regions.geojson \
            --match_type "region"

        python -m watch.cli.merge_region_models \
            --src $DVC_DPATH/drop1/region_models/*.geojson \
            --dst $DVC_DPATH/drop1/all_regions.geojson
    """


def main(cmdline=False, **kwargs):
    r"""

    CommandLine:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc \
            xdoctest -m watch.cli.merge_region_models main

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> import glob
        >>> dvc_repo = os.environ.get('DVC_DPATH')
        >>> region_fpath = os.path.join(dvc_repo, 'drop1/region_models')
        >>> json_paths = list(glob.glob(os.path.join(region_fpath, '*.geojson')))
        >>> kwargs = {'src': json_paths}
        >>> main(**kwargs)

    """
    config = MergeRegionModelConfig(default=kwargs, cmdline=cmdline)
    # print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    json_paths = config['src']
    output_fpath = config['dst']

    combo = combine_region_models(json_paths, config['match_type'])

    if output_fpath is None:
        print(geojson.dumps(combo, indent='    '))
    else:
        print('write to output_fpath = {!r}'.format(output_fpath))
        with open(output_fpath, 'w') as file:
            geojson.dump(combo, file, indent='    ')


_SubConfig = MergeRegionModelConfig


def combine_region_models(json_paths, match_type=None):
    if match_type is None:
        match_re = None
    else:
        match_re = re.compile(match_type)

    # Collect features with the "region" type
    all_region_features = []
    for json_fpath in json_paths:
        with open(json_fpath, 'r') as file:
            data = json.load(file)
        assert data['type'] == 'FeatureCollection'
        collection = geojson.FeatureCollection(**data)
        for feat in collection['features']:
            if match_re is None or match_re.match(feat['properties']['type']):
                all_region_features.append(feat)
                region_features.append(feat)
            if match_subre is not None and match_subre.match(feat['properties']['type']):
                site_features.append(feat)
        # Check if we have the expected no. of regions/sites
        if len(region_features) == 1:
            region_feature = region_features[0]
            region['properties']['sites'] = site_features
            all_region_features.append(region_feature)
        else:
            if len(site_features) > 0:
                print('WARNING: discarding sites belonging to ambiguous region')
            all_region_features.extend(region_features)

    combo = geojson.FeatureCollection(features=all_region_features)
    return combo


if __name__ == '__main__':
    """
    """
    main(cmdline=True)
