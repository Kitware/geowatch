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

    draw_clusters = scfg.Value(False, isflag=True, help='if True draw the clusters')

    crop_time = scfg.Value(True, isflag=True, help=ub.paragraph(
        '''
        if True also crops temporal extent to the sites, otherwise uses the region extent
        '''))

    minimum_size = scfg.Value('128x128@2GSD', help='minimum size of a cluster box')
    maximum_size = scfg.Value('1024x1024@2GSD', help='minimum size of a cluster box')

    context_factor = scfg.Value(1.5, help='extra padding around each site')


def main(cmdline=1, **kwargs):
    """
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
        >>> kwargs = {
        >>>     'src': src_fpath,
        >>>     'dst_dpath': dst_dpath,
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
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    import geopandas as gpd
    import kwimage
    from kwutil import util_time

    from geowatch import heuristics
    from geowatch.geoannots import geomodels
    from geowatch.utils import process_context
    from geowatch.utils import util_gis
    from geowatch.utils import util_kwimage
    from geowatch.utils import util_resolution

    # import pandas as pd
    if config.dst_dpath is None:
        raise ValueError('Destination path is required')

    dst_dpath = ub.Path(config.dst_dpath)
    rich.print(f'Will write to: [link={dst_dpath}]{dst_dpath}[/link]')

    proc_context = process_context.ProcessContext(
        name='geowatch.cli.cluster_sites', type='process',
        config=process_context.jsonify_config(config),
        track_emissions=False,
    )

    input_region_models = list(geomodels.RegionModel.coerce_multiple(
        config.src, workers=config.io_workers))

    if config.dst_region_fpath is not None:
        assert len(input_region_models) == 1, (
            'we assume only 1 input region when output fpath given')
        proc_context.start()

    scale = config.context_factor
    minimum_size = util_resolution.ResolvedWindow.coerce(config.minimum_size)
    maximum_size = util_resolution.ResolvedWindow.coerce(config.maximum_size)

    # Convert to meters
    meter = util_resolution.ResolvedUnit(1, 'GSD')
    min_box_dim = minimum_size.at_resolution(meter).window[0]
    max_box_dim = maximum_size.at_resolution(meter).window[0]

    print(f'Looping over {len(input_region_models)} region')

    all_final_site_summaries = []

    for input_region_model in input_region_models:

        input_region_model.fixup()

        if 1:
            try:
                input_region_model.validate(strict=0)
            except Exception:
                input_region_model.fixup()
                input_region_model.validate(strict=0)

        region_id = input_region_model.region_id
        region_header = input_region_model.pandas_region()
        region_sites = input_region_model.pandas_summaries()

        region_sites_utm = util_gis.project_gdf_to_local_utm(region_sites, mode=1, tolerance=None)

        polygons = kwimage.PolygonList([kwimage.Polygon.from_shapely(s) for s in region_sites_utm.geometry])

        # TODO: would be good to do this in 3D with temporal extent too.
        keep_bbs, overlap_idxs = util_kwimage.find_low_overlap_covering_boxes(
            polygons, scale, min_box_dim, max_box_dim, max_iters=100)

        assert len(region_header) == 1

        # for idxs in overlap_idxs:
        utm_crs = region_sites_utm.crs
        crs84 = region_header.crs

        subregion_ids = []
        subregion_suffix_list = []

        # Sort bbs so the largest spatial ones are first.
        keep_bbs = keep_bbs[keep_bbs.area.ravel().argsort()[::-1]]

        total_box = kwimage.Box.coerce(keep_bbs.bounding_box())
        total_geom = total_box.to_shapely()
        total_summaries = []

        total_end_date = None
        total_start_date = None

        print(f'Looping over {len(keep_bbs)} clusters')
        for bb_idx, utm_box in enumerate(keep_bbs.to_shapely()):

            ISECT_BOXES = 0
            if ISECT_BOXES:
                region_utm = region_header.to_crs(utm_crs)
                region_bounds = region_utm.iloc[0]['geometry']
                final_geom_utm = region_bounds.intersection(utm_box)
            else:
                final_geom_utm = utm_box

            final_geom_df_crs84 = gpd.GeoDataFrame({'geometry': [final_geom_utm]}, crs=utm_crs).to_crs(crs84)
            final_geom_crs84 = final_geom_df_crs84.iloc[0]['geometry']
            geometry = final_geom_crs84

            is_contained = region_sites_utm.intersects(final_geom_utm)
            contained_sites = region_sites[is_contained]

            # if 'predicted_phase_transition_date' in contained_sites:
            #     f = contained_sites['predicted_phase_transition_date'].isnull()
            #     tmp = contained_sites['predicted_phase_transition_date'].apply(str)
            #     tmp.loc[f[f].index] = None
            #     contained_sites = contained_sites.assign(predicted_phase_transition_date=tmp)
            # contained_sites = contained_sites.drop('region_id', axis=1)

            # contained_site_ids = sorted(contained_sites['site_id'].tolist())
            # contained_hashid = ub.hash_data(contained_site_ids, base=26)[0:8]
            # cluster_suffix = f'{bb_idx:03d}_n{len(contained_sites):03d}_{contained_hashid}'
            cluster_suffix = f'{bb_idx:03d}'

            subregion_id = f'{region_id}_CLUSTER_{cluster_suffix}'
            subregion_ids.append(subregion_id)
            subregion_suffix_list.append(cluster_suffix)

            start_dates = contained_sites['start_date'].dropna()
            end_dates = contained_sites['end_date'].dropna()

            # unused_cols = contained_sites.isna().all(axis=0)
            # Drop columns that dont go in site summaries
            # contained_sites = contained_sites.drop(['region_id', 'comments'], axis=1)

            site_summaries = list(geomodels.SiteSummary.from_geopandas_frame(contained_sites))

            if 1:
                for s in site_summaries:
                    s.fixup()
                    s.validate(strict=0)

            if len(start_dates) and config.crop_time:
                start_date = start_dates.min()
            else:
                start_date = input_region_model.start_date

            if len(end_dates) and config.crop_time:
                end_date = end_dates.max()
            else:
                end_date = input_region_model.end_date

            if geometry is None or geometry.is_empty or not geometry.is_valid:
                raise AssertionError

            _end_date = util_time.coerce_datetime(end_date)
            _start_date = util_time.coerce_datetime(start_date)
            if total_start_date is None:
                total_start_date = _start_date
            if total_end_date is None:
                total_end_date = _end_date

            sub_region_summary = geomodels.SiteSummary(
                properties={
                    "type": 'site_summary',
                    "status": "system_confirmed",
                    "site_id": subregion_id,
                    "version": '2.4.3',
                    "mgrs": None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "originator": "kit-cluster",
                    "model_content": "annotation",
                },
                geometry=geometry,
            )
            sub_region_summary.ensure_isodates()
            sub_region_summary.infer_mgrs()
            total_summaries.append(sub_region_summary)

            sub_region_header = geomodels.RegionHeader(
                properties={
                    "type": 'region',
                    "region_id": subregion_id,
                    "version": '2.4.3',
                    "mgrs": None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "originator": "kit-cluster",
                    "model_content": "annotation",
                    "comments": '',
                },
                geometry=geometry,
            )
            assert sub_region_header.geometry is not None
            sub_region_header.ensure_isodates()
            sub_region_header.infer_mgrs()

            sub_region = geomodels.RegionModel(features=[
                sub_region_header
            ] + site_summaries)
            all_final_site_summaries.extend(site_summaries)

            dst_dpath.ensuredir()
            fpath = dst_dpath / (subregion_id + '.geojson')

            try:
                sub_region.validate(strict=False)
            except Exception:
                sub_region.fixup()
                sub_region.validate(strict=False)

            fpath.write_text(sub_region.dumps())

        if config.dst_region_fpath is not None:
            total_geom_df_crs84 = gpd.GeoDataFrame({'geometry': [total_geom]}, crs=utm_crs).to_crs(crs84)
            total_geom_crs84 = total_geom_df_crs84.iloc[0]['geometry']

            obj = proc_context.stop()

            # TODO: use geomodels helper
            new_region_header = geomodels.RegionHeader(
                properties={
                    "type": 'region',
                    "region_id": region_id,
                    "version": '2.4.3',
                    "mgrs": None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "originator": "kit-cluster",
                    "model_content": "annotation",
                    "comments": '',
                    "cache": {
                        "process_context": obj,
                    }
                },
                geometry=total_geom_crs84,
            )
            new_region_header.ensure_isodates()
            new_region_header.infer_mgrs()
            new_region = geomodels.RegionModel(features=[
                new_region_header
            ] + total_summaries)
            region_fpath = ub.Path(config.dst_region_fpath)
            region_fpath.write_text(new_region.dumps())

        # print('all_final_site_summaries = {}'.format(ub.urepr(all_final_site_summaries, nl=1)))

        SHOW_SUBREGIONS = config.draw_clusters
        if SHOW_SUBREGIONS:
            status_list = region_sites_utm['status']
            color_list = []
            for status in status_list:
                info = heuristics.IARPA_STATUS_TO_INFO.get(status, {})
                color = kwimage.Color.coerce(info.get('color', 'pink')).as255()
                color_list.append(color)

            import kwplot
            from geowatch.utils import util_kwplot
            plt = kwplot.autoplt()
            kwplot.figure(fnum=1, doclf=1)
            # polygons.draw(color='pink')
            for poly, color in zip(polygons, color_list):
                edgecolor = kwimage.Color.coerce(color).adjust(saturate=-.1, lighten=.1)
                poly.draw(color=color, edgecolor=edgecolor, linewidth=1, alpha=0.7)
            # candidate_bbs.draw(color='blue', setlim=1)
            keep_bbs.draw(color='orange', setlim=1, labels=subregion_suffix_list)
            plt.gca().set_title('find_low_overlap_covering_boxes')
            fig = plt.gcf()
            viz_dpath = (dst_dpath / '_viz_clusters').ensuredir()
            import rich
            rich.print(f'Viz dpath: [link={viz_dpath}]{viz_dpath}[/link]')

            finalizer = util_kwplot.FigureFinalizer(
                size_inches=(16, 16),
                dpath=viz_dpath,
            )
            finalizer(fig, 'clusters_' + region_id + '.png')


__cli__ = ClusterSiteConfig
__cli__.main = main


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/cluster_sites.py
    """
    main(cmdline=True)
