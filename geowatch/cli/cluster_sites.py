#!/usr/bin/env python3
r"""
Given a set of site summaries, clusters them into groups, ideally with small
overlap. Writes new regions to a specified directory using the hash of the
contained sites as a subregion identifier.

Limitations:
    - The clustering algorithm is overly simple

Example:
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
    python -m geowatch.cli.cluster_sites \
            --src "$DVC_DATA_DPATH/annotations/drop6/region_models/KR_R002.geojson" \
            --dst_dpath $DVC_DATA_DPATH/ValiRegionSmall/geojson \
            --draw_clusters True

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
    python -m geowatch.cli.coco_align \
        --src $DVC_DATA_DPATH/Drop6/combo_imganns-KR_R002_L.kwcoco.json \
        --dst $DVC_DATA_DPATH/ValiRegionSmall/small_KR_R002_odarcigm.kwcoco.zip \
        --regions $DVC_DATA_DPATH/ValiRegionSmall/geojson/SUB_KR_R002_n007_odarcigm.geojson \
        --minimum_size="128x128@10GSD" \
        --context_factor=1 \
        --geo_preprop=auto \
        --force_nodata=-9999 \
        --site_summary=False \
        --target_gsd=5 \
        --aux_workers=8 \
        --workers=8

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
    python -m geowatch.cli.cluster_sites \
            --src "$DVC_DATA_DPATH/annotations/drop6/region_models/NZ_R001.geojson" \
            --dst_dpath $DVC_DATA_DPATH/ValiRegionSmall/geojson/NZ_R001 \
            --draw_clusters True

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
    python -m geowatch.cli.coco_align \
        --src $DVC_DATA_DPATH/Drop6/combo_imganns-NZ_R001_L.kwcoco.json \
        --dst $DVC_DATA_DPATH/ValiRegionSmall/small_NZ_R001_swnykmah.kwcoco.zip \
        --regions $DVC_DATA_DPATH/ValiRegionSmall/geojson/NZ_R001/SUB_NZ_R001_n031_swnykmah.geojson \
        --minimum_size="128x128@10GSD" \
        --context_factor=1 \
        --geo_preprop=auto \
        --force_nodata=-9999 \
        --site_summary=False \
        --target_gsd=5 \
        --aux_workers=8 \
        --workers=8
"""
import scriptconfig as scfg
import ubelt as ub


class ClusterSiteConfig(scfg.DataConfig):
    """
    Creates a new region file that groups nearby sites.
    """
    src = scfg.Value(None, help='input region files with site summaries', alias=['regions'])

    dst_dpath = scfg.Value(None, help='output path to store the resulting region files')

    dst_region_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        geojson output path for site summaries. Only works if there is a single
        input region, otherwise an error is raised
        '''))

    io_workers = scfg.Value(10, help='number of io workers')

    draw_clusters = scfg.Value(False, isflag=True, help='if True draw the clusters in the specified dst_dpath')

    crop_time = scfg.Value(True, isflag=True, help=ub.paragraph(
        '''
        if True also crops temporal extent to the sites, otherwise uses the region extent
        '''))

    minimum_size = scfg.Value('128x128@2GSD', help='minimum size of a cluster box')
    maximum_size = scfg.Value('1024x1024@2GSD', help='minimum size of a cluster box')

    context_factor = scfg.Value(1.5, help='extra padding around each site')

    ignore_status = scfg.Value("['system_rejected']", type=str, help=ub.paragraph(
        '''
        A YAML list of status values that should be ignored by the clustering
        algorithm.  Defaults to ["system_rejected"].
        '''))


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        xdoctest -m geowatch.cli.cluster_sites main:0
        xdoctest -m geowatch.cli.cluster_sites main:2

    Example:
        >>> from geowatch.cli.cluster_sites import *  # NOQA
        >>> from geowatch.cli import cluster_sites
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch', 'doctests', 'cluster_sites1').ensuredir()
        >>> src_dpath = (dpath / 'src').ensuredir()
        >>> dst_dpath = (dpath / 'dst')
        >>> dst_region_fpath = dst_dpath / 'cluster.geojson'
        >>> from geowatch.geoannots import geomodels
        >>> region = geomodels.RegionModel.random(num_sites=10)
        >>> src_fpath = src_dpath / 'demo_region.geojson'
        >>> src_fpath.write_text(region.dumps())
        >>> kwargs = {
        >>>     'src': src_fpath,
        >>>     'dst_dpath': dst_dpath,
        >>>     'dst_region_fpath': dst_region_fpath,
        >>>     'io_workers': 0,
        >>>     'draw_clusters': 0,
        >>>     'crop_time': True,
        >>> }
        >>> cmdline = 0
        >>> cluster_sites.main(cmdline=cmdline, **kwargs)

    Example:
        >>> from geowatch.cli.cluster_sites import *  # NOQA
        >>> from geowatch.cli import cluster_sites
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch', 'doctests', 'cluster_sites2').ensuredir()
        >>> src_dpath = (dpath / 'src').ensuredir()
        >>> dst_dpath = (dpath / 'dst')
        >>> from geowatch.geoannots import geomodels
        >>> region = geomodels.RegionModel.random(num_sites=10)
        >>> src_fpath = src_dpath / 'demo_region.geojson'
        >>> src_fpath.write_text(region.dumps())
        >>> dst_region_fpath = dst_dpath / 'cluster.geojson'
        >>> kwargs = {
        >>>     'src': src_fpath,
        >>>     'dst_dpath': dst_dpath,
        >>>     'dst_region_fpath': dst_region_fpath,
        >>>     'io_workers': 0,
        >>>     'draw_clusters': 1,
        >>>     'crop_time': True,
        >>> }
        >>> cmdline = 0
        >>> cluster_sites.main(cmdline=cmdline, **kwargs)

    Example:
        >>> # Test empty case
        >>> from geowatch.cli.cluster_sites import *  # NOQA
        >>> from geowatch.cli import cluster_sites
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch', 'doctests', 'cluster_sites3').ensuredir()
        >>> src_dpath = (dpath / 'src').ensuredir()
        >>> dst_dpath = (dpath / 'dst')
        >>> from geowatch.geoannots import geomodels
        >>> region = geomodels.RegionModel.random(num_sites=0)
        >>> src_fpath = src_dpath / 'demo_region.geojson'
        >>> src_fpath.write_text(region.dumps())
        >>> dst_region_fpath = dst_dpath / 'cluster.geojson'
        >>> kwargs = {
        >>>     'src': src_fpath,
        >>>     'dst_dpath': dst_dpath,
        >>>     'dst_region_fpath': dst_region_fpath,
        >>>     'io_workers': 0,
        >>>     'draw_clusters': 1,
        >>>     'crop_time': True,
        >>> }
        >>> cmdline = 0
        >>> cluster_sites.main(cmdline=cmdline, **kwargs)

    Ignore:
        import xdev
        xdev.profile_now(cluster_sites.main)(cmdline=cmdline, **kwargs)

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch'))
        from geowatch.cli.cluster_sites import *  # NOQA
        import geowatch
        data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        src = data_dpath / 'annotations/drop6/region_models/KR_R002.geojson'
        dst_dpath = data_dpath / 'ValiRegionSmall/geojson'
        kwargs = dict(src=src, dst_dpath=dst_dpath, draw_clusters=True)
        main(**kwargs)
    """
    config = ClusterSiteConfig.cli(cmdline=cmdline, data=kwargs)
    import rich
    from rich.markup import escape
    rich.print('config = {}'.format(escape(ub.urepr(config, nl=1))))

    from kwutil import util_yaml
    from geowatch.geoannots import geomodels
    from geowatch.utils import process_context
    from geowatch.utils import util_resolution

    config.ignore_status = util_yaml.Yaml.coerce(config.ignore_status)
    print(f'config.ignore_status = {ub.urepr(config.ignore_status, nl=1)}')

    # import pandas as pd
    if config.dst_dpath is None:
        raise ValueError('Destination path is required')

    dst_dpath = config.dst_dpath = ub.Path(config.dst_dpath)
    config.dst_dpath.ensuredir()
    rich.print(f'Will write to: [link={dst_dpath}]{dst_dpath}[/link]')

    proc_context = process_context.ProcessContext(
        name='geowatch.cli.cluster_sites', type='process',
        config=process_context.jsonify_config(config),
        track_emissions=False,
    )

    input_region_models = list(geomodels.RegionModel.coerce_multiple(
        config.src, workers=config.io_workers))

    if config.dst_region_fpath is not None:
        if len(input_region_models) != 1:
            raise ValueError(
                'we assume only 1 input region when output fpath given')
        # Fixme: process context is weird here when we allow for multiple
        # region inputs
        proc_context.start()

    scale = config.context_factor
    minimum_size = util_resolution.ResolvedWindow.coerce(config.minimum_size)
    maximum_size = util_resolution.ResolvedWindow.coerce(config.maximum_size)

    # Convert to meters
    meter = util_resolution.ResolvedUnit(1, 'GSD')
    min_box_dim = minimum_size.at_resolution(meter).window[0]
    max_box_dim = maximum_size.at_resolution(meter).window[0]

    print(f'Looping over {len(input_region_models)} region')

    # all_final_site_summaries = []

    # Note: this should only ever be a single item in this loop.  outputs will
    # clobber each other. The code is setup to allow flexability to draw
    # multiple region clusters at once, but we might want to disable this.
    for input_region_model in input_region_models:
        cluster_single_region_sites(input_region_model, scale, min_box_dim,
                                    max_box_dim, proc_context, config)


def cluster_single_region_sites(input_region_model, scale, min_box_dim, max_box_dim, proc_context, config):
    import kwimage
    from kwgis.utils import util_gis
    from geowatch.utils import util_kwimage
    input_region_model.fixup()

    if 1:
        try:
            input_region_model.validate(strict=0)
        except Exception:
            input_region_model.fixup()
            input_region_model.validate(strict=0)

    # Create the set of input boxes to the clustering algorithm.
    # Filter out any that have an ignorable status, and convert to UTM
    # coordinates.
    region_header = input_region_model.pandas_region()
    region_sites = input_region_model.pandas_summaries()
    assert len(region_header) == 1
    if config.ignore_status:
        # Remove any sites that have a status marked as ignored.
        if 'status' in region_sites.columns:
            keep_flags = region_sites['status'].apply(lambda s: s not in config.ignore_status)
            # For some reason this returns a dataframe instead of a
            # geodataframe if all flags are false Using an index with loc seems
            # to work around this.
            keep_indexes = keep_flags.index[keep_flags.index]
            region_sites = region_sites.loc[keep_indexes]

    region_sites_utm = util_gis.project_gdf_to_local_utm(region_sites, mode=1, tolerance=None)
    polygons = kwimage.PolygonList([kwimage.Polygon.from_shapely(s)
                                    for s in region_sites_utm.geometry])

    # Run the clustering algorithm.
    # TODO: would be good to do this in 3D with temporal extent too.
    keep_bbs, overlap_idxs = util_kwimage.find_low_overlap_covering_boxes(
        polygons, scale, min_box_dim, max_box_dim, max_iters=100)

    # Sort bbs so the largest spatial ones are first.
    keep_bbs = keep_bbs[keep_bbs.area.ravel().argsort()[::-1]]

    # Convert the UTM bounding boxes into proper region models.
    clustered_region, subregions = build_clustered_models(
        input_region_model, region_header, region_sites, region_sites_utm,
        keep_bbs, config)

    # Write output to disk
    # Awkward backwards compat, dump individal cluster groups
    for subregion in subregions:
        fpath = config.dst_dpath / (subregion.region_id + '.geojson')
        fpath.write_text(subregion.dumps())

    if config.dst_region_fpath is not None:
        obj = proc_context.stop()
        clustered_region.header['properties']['cache']['proc_context'] = obj
        region_fpath = ub.Path(config.dst_region_fpath)
        region_fpath.write_text(clustered_region.dumps())

    if config.draw_clusters:
        _draw_clusters(input_region_model, region_sites_utm, polygons,
                       clustered_region, keep_bbs, config)


def _draw_clusters(input_region_model, region_sites_utm, polygons,
                   clustered_region, keep_bbs, config):
    from geowatch import heuristics
    from geowatch.utils import util_kwplot
    import kwimage
    import kwplot
    import rich

    if 'status' in region_sites_utm.columns:
        color_list = []
        status_list = region_sites_utm['status']
        for status in status_list:
            info = heuristics.IARPA_STATUS_TO_INFO.get(status, {})
            color = kwimage.Color.coerce(info.get('color', 'pink')).as255()
            color_list.append(color)
    else:
        color_list = ['blue'] * len(polygons)

    plt = kwplot.autoplt()
    kwplot.figure(fnum=1, doclf=1)
    # polygons.draw(color='pink')
    for poly, color in zip(polygons, color_list):
        edgecolor = kwimage.Color.coerce(color).adjust(saturate=-.1, lighten=.1)
        poly.draw(color=color, edgecolor=edgecolor, linewidth=1, alpha=0.7)
    # candidate_bbs.draw(color='blue', setlim=1)
    subregion_suffix_list = [
        ss['properties']['cache']['cluster_suffix']
        for ss in clustered_region.site_summaries()
    ]
    if len(keep_bbs):
        keep_bbs.draw(color='orange', setlim=1, labels=subregion_suffix_list)
    plt.gca().set_title('find_low_overlap_covering_boxes')
    fig = plt.gcf()
    viz_dpath = (config.dst_dpath / '_viz_clusters').ensuredir()
    rich.print(f'Viz dpath: [link={viz_dpath}]{viz_dpath}[/link]')

    finalizer = util_kwplot.FigureFinalizer(
        size_inches=(16, 16),
        dpath=viz_dpath,
    )
    finalizer(fig, 'clusters_' + input_region_model.region_id + '.png')


def build_clustered_models(input_region_model, region_header, region_sites,
                           region_sites_utm, keep_bbs, config):
    """
    Given the clustering output, construct new region models.

    Returns:
        Tuple[RegionModel, List[RegionModle]] :

        1. A new "clustered region" region model with the original region
           bounds and the clusters as site summaries

        2. A set of region models for each cluster, with the original site
           summaries that were assigned to that cluster

    """
    from geowatch.geoannots import geomodels
    import geopandas as gpd

    region_id = input_region_model.region_id

    utm_crs = region_sites_utm.crs
    crs84 = region_header.crs

    # Start an empty region model with the original bounds
    # we will add clusters as site summaries to this data structure.
    new_region_header = geomodels.RegionHeader(
        properties={
            "type": 'region',
            "region_id": input_region_model.region_id,
            "version": '2.4.3',
            "mgrs": None,
            "start_date": input_region_model.start_date.date().isoformat(),
            "end_date": input_region_model.end_date.date().isoformat(),
            "originator": "kit-cluster",
            "model_content": "annotation",
            "comments": '',
            "cache": {}
        },
        geometry=input_region_model.geometry,
    )
    new_region_header.ensure_isodates()
    new_region_header.infer_mgrs()
    clustered_region = geomodels.RegionModel(features=[new_region_header])

    print(f'Looping over {len(keep_bbs)} clusters')
    subregions = []
    for bb_idx, utm_box in enumerate(keep_bbs.to_shapely()):

        ISECT_BOXES = 0
        if ISECT_BOXES:
            region_utm = region_header.to_crs(utm_crs)
            region_bounds = region_utm.iloc[0]['geometry']
            final_geom_utm = region_bounds.intersection(utm_box)
        else:
            final_geom_utm = utm_box

        # Convert the UTM cluster into CRS84
        final_geom_df_crs84 = gpd.GeoDataFrame(
            {'geometry': [final_geom_utm]}, crs=utm_crs).to_crs(crs84)
        final_geom_crs84 = final_geom_df_crs84.iloc[0]['geometry']
        cluster_geometry = final_geom_crs84
        if (cluster_geometry is None or
             cluster_geometry.is_empty or
             not cluster_geometry.is_valid):
            raise AssertionError

        # Assign original sites to this cluster.
        is_contained = region_sites_utm.intersects(final_geom_utm)
        contained_sites = region_sites[is_contained]

        cluster_suffix = f'{bb_idx:03d}'
        cluster_id = f'{region_id}_CLUSTER_{cluster_suffix}'
        start_dates = contained_sites['start_date'].dropna()
        end_dates = contained_sites['end_date'].dropna()

        site_summaries = list(geomodels.SiteSummary.from_geopandas_frame(contained_sites))
        if 1:
            for s in site_summaries:
                s.fixup()
                s.validate(strict=0)

        # The start / end date of the cluster is determined by the original
        # sites that intersect it.
        if len(start_dates) and config.crop_time:
            start_date = start_dates.min()
        else:
            start_date = input_region_model.start_date
        if len(end_dates) and config.crop_time:
            end_date = end_dates.max()
        else:
            end_date = input_region_model.end_date

        # Add this cluster as a site summary in the new region
        cluster_summary = geomodels.SiteSummary(
            properties={
                "type": 'site_summary',
                "status": "system_confirmed",
                "site_id": cluster_id,
                "version": '2.4.3',
                "mgrs": None,
                "start_date": start_date,
                "end_date": end_date,
                "originator": "kit-cluster",
                "model_content": "annotation",
                "cache": {"cluster_suffix": cluster_suffix},
            },
            geometry=cluster_geometry,
        )
        cluster_summary.ensure_isodates()
        cluster_summary.infer_mgrs()
        clustered_region.add_site_summary(cluster_summary)

        # Create a new region model where the cluster is the region boundary
        # that contatain the original site that were assigned to it.
        subregion_header = geomodels.RegionHeader(
            properties={
                "type": 'region',
                "region_id": cluster_id,
                "version": '2.4.3',
                "mgrs": None,
                "start_date": start_date,
                "end_date": end_date,
                "originator": "kit-cluster",
                "model_content": "annotation",
                "comments": '',
            },
            geometry=cluster_geometry,
        )
        assert subregion_header.geometry is not None
        subregion_header.ensure_isodates()
        subregion_header.infer_mgrs()
        subregion = geomodels.RegionModel(features=[
            subregion_header
        ] + site_summaries)
        try:
            subregion.validate(strict=False)
        except Exception:
            subregion.fixup()
            subregion.validate(strict=False)
        subregions.append(subregion)

    return clustered_region, subregions


__cli__ = ClusterSiteConfig
__cli__.main = main


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/cluster_sites.py
    """
    main(cmdline=True)
