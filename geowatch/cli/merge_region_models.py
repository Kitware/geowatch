"""
"""
import re
import ubelt as ub
import scriptconfig as scfg
import json


class MergeRegionModelConfig(scfg.DataConfig):
    """
    Combine the specific features from multiple region files into a single one.
    """
    __default__ = {
        'src': scfg.Value([], nargs='+', position=1,
                          help='paths to input geojson region files'),

        'dst': scfg.Value(None, help='path to combined multi-region file'),

        'match_type': scfg.Value('region', help=ub.paragraph(
            '''
            regex that filters results to only contain features where
            properties.type match.
            ''')),

        'match_subtype': scfg.Value('site', help=ub.paragraph(
            '''
            regex that filters results to include subregion features as
            metadata inside regions, where properties.type match.
            ''')),
    }
    epilog = r"""
    Example Usage:

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

        python -m geowatch merge_region_models \
            --src $DVC_DPATH/drop1/region_models/*.geojson \
            --dst $DVC_DPATH/drop1/all_regions.geojson \
            --match_type "region"

        python -m geowatch.cli.merge_region_models \
            --src $DVC_DPATH/drop1/region_models/*.geojson \
            --dst $DVC_DPATH/drop1/all_regions.geojson
    """


def main(cmdline=False, **kwargs):
    r"""

    CommandLine:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc \
            xdoctest -m geowatch.cli.merge_region_models main

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
    import geojson
    config = MergeRegionModelConfig.cli(data=kwargs, cmdline=cmdline, strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1)))

    json_paths = config['src']
    output_fpath = config['dst']

    combo = combine_region_models(json_paths, config['match_type'],
                                  config['match_subtype'])

    if output_fpath is None:
        print(geojson.dumps(combo, indent='    '))
    else:
        print('write to output_fpath = {!r}'.format(output_fpath))
        with open(output_fpath, 'w') as file:
            geojson.dump(combo, file, indent='    ')


__config__ = MergeRegionModelConfig


def combine_region_models(json_paths, match_type=None, match_subtype=None):
    import geojson
    if match_type is None:
        match_re = None
    else:
        match_re = re.compile(match_type)

    if match_subtype is None:
        match_subre = None
    else:
        match_subre = re.compile(match_subtype)

    # Collect features with the "region" type
    all_region_features = []
    for json_fpath in json_paths:
        # Inside each file, there should be one "region" and several "site"s
        # Save the sites inside the region
        with open(json_fpath, 'r') as file:
            data = json.load(file)
        assert data['type'] == 'FeatureCollection'
        collection = geojson.FeatureCollection(**data)
        region_features = []
        site_features = []
        for feat in collection['features']:
            if match_re is None or match_re.match(feat['properties']['type']):
                region_features.append(feat)
            if match_subre is not None and match_subre.match(
                    feat['properties']['type']):
                site_features.append(feat)
        # Check if we have the expected no. of regions/sites
        if len(region_features) == 1:
            region_feature = region_features[0]
            region_feature['properties']['sites'] = site_features
            all_region_features.append(region_feature)
        else:
            if len(site_features) > 0:
                print(
                    'WARNING: discarding sites belonging to ambiguous region')
            all_region_features.extend(region_features)

    combo = geojson.FeatureCollection(features=all_region_features)
    return combo


if __name__ == '__main__':
    """
    """
    main(cmdline=True)
