#!/usr/bin/env python
r"""
Assigns geospace annotation to image pixels and frames

CommandLine:
    # Update a dataset with new annotations
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

    # You dont need to run this, but if you plan on running the project
    # annotations script multiple times, preloading this work will make it
    # faster
    python -m watch add_fields \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
        --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
        --overwrite=warp --workers 10

    # Update to whatever the state of the annotations submodule is
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    python -m watch project_annotations \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --viz_dpath $DVC_DPATH/Drop2-Aligned-TA1-2022-02/_viz_propogate \
        --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

    python -m watch visualize \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
        --space="video" \
        --num_workers=avail \
        --any3="only" --draw_anns=True --draw_imgs=False --animate=True
"""
import dateutil
import kwcoco
import kwimage
import ubelt as ub
import numpy as np
import scriptconfig as scfg
from watch.utils import kwcoco_extensions
from watch.utils import util_kwplot
from watch.utils import util_path
from watch.utils import util_time


class ProjectAnnotationsConfig(scfg.Config):
    """
    Projects annotations from geospace onto a kwcoco dataset and optionally
    propogates them in

    References:
        https://smartgitlab.com/TE/annotations/-/wikis/Alternate-Site-Type
    """
    default = {
        'src': scfg.Value(help=ub.paragraph(
            '''
            path to the kwcoco file to propagate labels in
            '''), position=1),

        'dst': scfg.Value('propagated_data.kwcoco.json', help=ub.paragraph(
            '''
            Where the output kwcoco file with propagated labels is saved
            '''), position=2),

        'site_models': scfg.Value(None, help=ub.paragraph(
            '''
            Geospatial geojson annotation files. Either a path to a file, or a
            directory.
            ''')),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            if specified, visualizations will be written to this directory
            ''')),

        'verbose': scfg.Value(1, help=ub.paragraph(
            '''
            use this to print details
            ''')),

        'clear_existing': scfg.Value(True, help=ub.paragraph(
            '''
            if True, clears existing annotations before projecting the new ones.
            ''')),

        'propogate': scfg.Value(True, help='if True does forward propogation in time'),

        'geo_preprop': scfg.Value('auto', help='force if we check geo properties or not'),

        'geospace_lookup': scfg.Value('auto', help='if False assumes region-ids can be used to lookup association'),

        'workers': scfg.Value(0, help='number of workers for geo-preprop if done'),
    }


def main(cmdline=False, **kwargs):
    """
    CommandLine:
        python -m watch.cli.project_annotations \
            --src

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.cli.project_annotations import *  # NOQA
        >>> from watch.utils import util_data
        >>> import tempfile
        >>> dvc_dpath = util_data.find_smart_dvc_dpath()
        >>> bundle_dpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/'
        >>> dpath = ub.Path(ub.ensure_app_cache_dir('watch/tests/project_annots'))
        >>> cmdline = False
        >>> output_fpath = dpath / 'data.kwcoco.json'
        >>> viz_dpath = (dpath / 'viz').ensuredir()
        >>> kwargs = {
        >>>     'src': bundle_dpath / 'data.kwcoco.json',
        >>>     'dst': output_fpath,
        >>>     'viz_dpath': viz_dpath,
        >>>     'site_models': dvc_dpath / 'drop1/site_models',
        >>> }
        >>> main(**kwargs)
    """
    import geopandas as gpd  # NOQA
    config = ProjectAnnotationsConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    output_fpath = config['dst']
    if output_fpath is None:
        raise AssertionError

    # Load the coco dataset with all of the images
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    site_geojson_fpaths = util_path.coerce_patterned_paths(config['site_models'], '.geojson')

    geo_preprop = config['geo_preprop']
    geospace_lookup = config['geospace_lookup']
    if geo_preprop == 'auto':
        coco_img = coco_dset.coco_image(ub.peek(coco_dset.index.imgs.keys()))
        geo_preprop = not any('geos_corners' in obj for obj in coco_img.iter_asset_objs())
        print('auto-choose geo_preprop = {!r}'.format(geo_preprop))

    if geo_preprop:
        kwcoco_extensions.coco_populate_geo_heuristics(
            coco_dset, overwrite={'warp'}, workers=config['workers'],
            keep_geotiff_metadata=False,
        )

    # Read the external CRS84 annotations from the site models
    sites = []
    from watch.utils import util_gis
    for fpath in ub.ProgIter(site_geojson_fpaths, desc='load geojson annots'):
        gdf = util_gis.read_geojson(fpath)
        sites.append(gdf)

    if config['clear_existing']:
        coco_dset.clear_annotations()

    propogate = config['propogate']
    propogated_annotations, all_drawable_infos = assign_sites_to_images(
        coco_dset, sites, propogate, geospace_lookup=geospace_lookup)

    for ann in propogated_annotations:
        coco_dset.add_annotation(**ann)
    kwcoco_extensions.warp_annot_segmentations_from_geos(coco_dset)

    coco_dset.fpath = output_fpath
    print('dump coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    coco_dset.dump(coco_dset.fpath)

    viz_dpath = config['viz_dpath']
    if viz_dpath:
        import kwplot
        kwplot.autoplt()
        viz_dpath = ub.Path(viz_dpath).ensuredir()
        for fnum, info in enumerate(ub.ProgIter(all_drawable_infos, desc='draw region site propogation', verbose=3)):
            drawable_region_sites = info['drawable_region_sites']
            region_id = info['region_id']
            region_image_dates = info['region_image_dates']
            fig = kwplot.figure(fnum=fnum)
            ax = fig.gca()
            plot_image_and_site_times(coco_dset, region_image_dates,
                                      drawable_region_sites, region_id, ax=ax)

            fig.set_size_inches(np.array([16, 9]) * 1)
            plot_fpath = viz_dpath / f'time_propogation_{region_id}.png'
            fig.tight_layout()
            fig.savefig(plot_fpath)


def assign_sites_to_images(coco_dset, sites, propogate, geospace_lookup='auto'):
    """
    Given a coco dataset (with geo information) and a list of geojson sites,
    determines which images each site-annotations should go on.
    """
    from shapely.ops import unary_union
    import pandas as pd
    from watch.utils import util_gis
    # Create a geopandas data frame that contains the CRS84 extent of all images
    img_gdf = kwcoco_extensions.covered_image_geo_regions(coco_dset)

    # Group all images by their video-id and take a union of their geometry to
    # get a the region covered by the video.
    video_gdfs = []
    vidid_to_imgdf = {}
    for vidid, subdf in img_gdf.groupby('video_id'):
        subdf = subdf.sort_values('frame_index')
        video_gdf = subdf.dissolve()
        video_gdf = video_gdf.drop(['date_captured', 'name', 'image_id', 'frame_index', 'height', 'width'], axis=1)
        combined = unary_union(list(subdf.geometry.values))
        video_gdf['geometry'].iloc[0] = combined
        video_gdfs.append(video_gdf)
        vidid_to_imgdf[vidid] = subdf
    videos_gdf = pd.concat(video_gdfs)

    PROJECT_ENDSTATE = True

    region_id_to_sites = ub.group_items(sites, lambda x: x.iloc[0]['region_id'])

    if 0:
        site_high_level_summaries = []
        for region_id, region_sites in region_id_to_sites.items():
            print('=== {} ==='.format(region_id))
            for site_gdf in region_sites:
                site_summary_row = site_gdf.iloc[0]
                site_rows = site_gdf.iloc[1:]
                track_id = site_summary_row['site_id']
                status = site_summary_row['status']
                summary = {
                    'region_id': region_id,
                    'track_id': track_id,
                    'status': status,
                    'start_date': site_summary_row['start_date'],
                    'end_date': site_summary_row['end_date'],
                    'unique_phases': site_rows['current_phase'].unique(),
                }
                print('summary = {}'.format(ub.repr2(summary, nl=1)))
                site_high_level_summaries.append(summary)

        df = pd.DataFrame(site_high_level_summaries)
        for region_id, subdf in df.groupby('region_id'):
            print('=== {} ==='.format(region_id))
            subdf = subdf.sort_values('status')
            print(subdf.to_string())

    # Ensure colors and categories
    from watch import heuristics
    status_to_color = {d['name']: kwimage.Color(d['color']).as01()
                       for d in heuristics.HUERISTIC_STATUS_DATA}
    print(coco_dset.dataset['categories'])
    for cat in heuristics.CATEGORIES:
        coco_dset.ensure_category(**cat)
    # hack in heuristic colors
    heuristics.ensure_heuristic_coco_colors(coco_dset)
    # handle any other colors
    kwcoco_extensions.category_category_colors(coco_dset)
    print(coco_dset.dataset['categories'])

    all_drawable_infos = []  # helper if we are going to draw

    if geospace_lookup == 'auto':
        coco_video_names = set(coco_dset.index.name_to_video.keys())
        region_ids = set(region_id_to_sites.keys())
        geospace_lookup = not coco_video_names.issubset(region_ids)
        print('geospace_lookup = {!r}'.format(geospace_lookup))

    def coerce_datetime2(data):
        """ Is this a monad 🦋 ? """
        if data is None:
            return data
        else:
            return util_time.coerce_datetime(data)

    propogated_annotations = []
    for region_id, region_sites in region_id_to_sites.items():

        # Find the video associated with this region
        # If this assumption is not valid, we could refactor to loop through
        # each site, do the geospatial lookup, etc...
        # but this is faster if we know regions are consistent
        if not geospace_lookup:
            try:
                video = coco_dset.index.name_to_video[region_id]
            except KeyError:
                print('No region-id match for region_id={}'.format(region_id))
                continue
            video_id = video['id']
            # video_name = video['name']
        else:
            video_ids = []
            for site_gdf in region_sites:
                # determine which video it the site belongs to
                video_overlaps = util_gis.geopandas_pairwise_overlaps(site_gdf, videos_gdf)
                overlapping_video_indexes = set(np.hstack(list(video_overlaps.values())))
                if len(overlapping_video_indexes) > 0:
                    assert len(overlapping_video_indexes) == 1, 'should only belong to one video'
                    overlapping_video_index = ub.peek(overlapping_video_indexes)
                    video_id = videos_gdf.iloc[overlapping_video_index]['video_id']
                    # video_name = coco_dset.index.videos[video_id]['name']
                    # assert site_gdf.iloc[0].region_id == video_name, 'sanity check'
                    # assert site_gdf.iloc[0].region_id == region_id, 'sanity check'
                    video_ids.append(video_id)
            assert ub.allsame(video_ids)
            if len(video_ids) == 0:
                print('No geo-space match for region_id={}'.format(region_id))
                continue
            video_id = video_ids[0]

        # Grab the images data frame for that video
        subimg_df = vidid_to_imgdf[video_id]
        region_image_dates = np.array(list(map(dateutil.parser.parse, subimg_df['date_captured'])))
        region_image_indexes = np.arange(len(region_image_dates))
        region_gids = subimg_df['image_id'].values

        drawable_region_sites = []

        # For each site in this region
        for site_gdf in region_sites:
            if __debug__:
                # Sanity check, the sites should have spatial overlap with each image in the video
                image_overlaps = util_gis.geopandas_pairwise_overlaps(site_gdf, subimg_df)
                num_unique_overlap_frames = set(ub.map_vals(len, image_overlaps).values())
                assert len(num_unique_overlap_frames) == 1

            site_summary_row = site_gdf.iloc[0]
            site_rows = site_gdf.iloc[1:]
            track_id = site_summary_row['site_id']
            status = site_summary_row['status']

            start_date = coerce_datetime2(site_summary_row['start_date'])
            end_date = coerce_datetime2(site_summary_row['end_date'])

            flags = ~site_rows['observation_date'].isnull()
            valid_site_rows = site_rows[flags]

            observation_dates = np.array([
                coerce_datetime2(x) for x in valid_site_rows['observation_date']
            ])

            if start_date is not None and observation_dates[0] != start_date:
                raise AssertionError
            if end_date is not None and observation_dates[-1] != end_date:
                raise AssertionError

            # Determine the first image each site-observation will be
            # associated with and then propogate them forward as necessary.
            try:
                found_idxs = np.searchsorted(region_image_dates, observation_dates, 'left')
            except TypeError:
                # handle  can't compare offset-naive and offset-aware datetimes
                region_image_dates = [util_time.ensure_timezone(dt)
                                      for dt in region_image_dates]
                observation_dates = [util_time.ensure_timezone(dt)
                                     for dt in observation_dates]
                found_idxs = np.searchsorted(region_image_dates, observation_dates, 'left')

            image_idxs_per_observation = np.split(region_image_indexes, found_idxs)[1:]

            # Create annotations on each frame we are associated with
            site_anns = []
            drawable_summary = []
            for gxs, site_row in zip(image_idxs_per_observation, site_rows.to_dict(orient='records')):
                site_row['geometry']
                gids = region_gids[gxs]
                site_row_datetime = coerce_datetime2(site_row['observation_date'])
                assert site_row_datetime is not None

                catname = site_row['current_phase']
                if catname is None:
                    # Based on the status choose a kwcoco category name
                    # using the watch heuristics
                    catname = heuristics.PHASE_STATUS_TO_KWCOCO_CATNAME[status]

                propogated_on = []
                category_colors = []
                categories = []

                # Handle multi-category per-row logic
                site_catnames = [c.strip() for c in catname.split(',')]
                row_summary = {
                    'track_id': track_id,
                    'site_row_datetime': site_row_datetime,
                    'propogated_on': propogated_on,
                    'category_colors': category_colors,
                    'categories': categories,
                    'status': status,
                    'color': status_to_color[status],
                }

                site_polygons = [
                    p.to_geojson()
                    for p in kwimage.MultiPolygon.from_shapely(site_row['geometry']).to_multi_polygon().data
                ]
                assert len(site_polygons) == len(site_catnames)

                for gid in gids:
                    img = coco_dset.imgs[gid]
                    img_datetime = util_time.coerce_datetime(img['date_captured'])
                    if propogate or img_datetime == site_row_datetime:
                        hack = 0
                        for catname, poly in zip(site_catnames, site_polygons):
                            # TODO: use heuristic module
                            if catname == 'Post Construction':
                                # Don't project end-states of we dont want to
                                if not PROJECT_ENDSTATE:
                                    continue
                            if hack == 0:
                                propogated_on.append(img_datetime)
                                hack = 1
                            cid = coco_dset.ensure_category(catname)
                            cat = coco_dset.index.cats[cid]
                            category_colors.append(cat['color'])
                            categories.append(catname)
                            img['date_captured']
                            ann = {
                                'image_id': gid,
                                'segmentation_geos': poly,
                                'status': status,
                                'category_id': cid,
                                'track_id': track_id,
                            }
                            site_anns.append(ann)
                drawable_summary.append(row_summary)
            propogated_annotations.extend(site_anns)
            drawable_region_sites.append(drawable_summary)

        drawable_region_sites = sorted(
            drawable_region_sites,
            key=lambda drawable_summary: min([r['site_row_datetime'] for r in drawable_summary]))

        all_drawable_infos.append({
            'drawable_region_sites': drawable_region_sites,
            'region_id': region_id,
            'region_image_dates': region_image_dates,
        })

    return propogated_annotations, all_drawable_infos


def plot_image_and_site_times(coco_dset, region_image_dates, drawable_region_sites, region_id, ax=None):
    """
    References:
        .. [HandleDates] https://stackoverflow.com/questions/44642966/how-to-plot-multi-color-line-if-x-axis-is-date-time-index-of-pandas
    """
    import kwplot
    if ax is None:
        plt = kwplot.autoplt()
        ax = plt.gca()

    ax.cla()

    from watch.tasks.fusion import heuristics
    import matplotlib as mpl
    hueristic_status_data = heuristics.HUERISTIC_STATUS_DATA

    status_to_color = {d['name']: kwimage.Color(d['color']).as01()
                       for d in hueristic_status_data}
    # region_status_labels = {site_gdf.iloc[0]['status'] for site_gdf in region_sites}
    # for status in region_status_labels:
    #     if status not in status_to_color:
    #         hueristic_status_data.append({
    #             'name': status,
    #             'color': kwimage.Color.random().as01(),
    #         })
    # status_to_color = {d['name']: kwimage.Color(d['color']).as01()
    #                    for d in hueristic_status_data}

    bounds_segments = []
    for t in ub.ProgIter(region_image_dates, desc='plot bounds'):
        y2 = len(drawable_region_sites) + 1
        x1 = mpl.dates.date2num(t)
        xy1 = (x1, 0)
        xy2 = (x1, y2)
        segment = [xy1, xy2]
        bounds_segments.append(segment)
        # ax.plot([t, t], [0, y2], color='darkblue', alpha=0.5)
    line_group = mpl.collections.LineCollection(
        bounds_segments, color='darkblue', alpha=0.5)
    ax.add_collection(line_group)

    if 1:
        ax.autoscale_view()
        ax.xaxis_date()

    all_times = []

    propogate_attrs = {
        'segments': [],
        'colors': [],
    }
    for summary_idx, drawable_summary in enumerate(ub.ProgIter(drawable_region_sites, desc='plot region')):
        site_dates = [r['site_row_datetime'] for r in drawable_summary]
        all_times.extend(site_dates)
        yloc = summary_idx
        status_color = drawable_summary[0]['color']

        # Draw propogations
        for row in drawable_summary:
            t1 = row['site_row_datetime']
            cat_colors = row['category_colors']
            yoffsets = np.linspace(0.5, 0.75, len(cat_colors))[::-1]

            # Draw a line for each "part" of the side at this timestep
            # Note: some sites seem to have a ton of parts that could be
            # consolodated? Is this real or is there a bug?
            for yoff, color in zip(yoffsets, cat_colors):
                for tp in row['propogated_on']:
                    x1 = mpl.dates.date2num(t1)
                    x2 = mpl.dates.date2num(tp)
                    y1 = yloc
                    y2 = yloc + yoff
                    segment = [(x1, y1), (x2, y2)]
                    propogate_attrs['segments'].append(segment)
                    propogate_attrs['colors'].append(color)
                    # ax.plot([x1, x2], [y1, y2], '-', color=color)

        # Draw site keyframes
        ax.plot(site_dates, [yloc] * len(site_dates), '-o', color=status_color, alpha=0.5)

    propogate_group = mpl.collections.LineCollection(
        propogate_attrs['segments'],
        color=propogate_attrs['colors'],
        alpha=0.5)
    ax.add_collection(propogate_group)

    propogate_attrs['segments']
    propogate_attrs['colors']

    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(0, len(drawable_region_sites))

    cat_to_color = {cat['name']: cat['color'] for cat in coco_dset.cats.values()}

    util_kwplot.phantom_legend(status_to_color, ax=ax, legend_id=1, loc=0)
    util_kwplot.phantom_legend(cat_to_color, ax=ax, legend_id=3, loc=3)

    ax.set_xlabel('Time')
    ax.set_ylabel('Site Index')
    ax.set_title('Site & Image Timeline: ' + region_id)
    return ax


def draw_geospace(dvc_dpath, sites):
    from watch.utils import util_gis
    import geopandas as gpd
    import kwplot
    kwplot.autompl()
    region_fpaths = util_path.coerce_patterned_paths(dvc_dpath / 'drop1/region_models', '.geojson')
    regions = []
    for fpath in ub.ProgIter(region_fpaths, desc='load geojson annots'):
        gdf = util_gis.read_geojson(fpath)
        regions.append(gdf)

    wld_map_gdf = gpd.read_file(
        gpd.datasets.get_path('naturalearth_lowres')
    )
    ax = wld_map_gdf.plot()

    for gdf in regions:
        centroid = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
        centroid.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)
        gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)

    for gdf in sites:
        centroid = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
        centroid.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)
        gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)


_SubConfig = ProjectAnnotationsConfig


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/project_annotations.py
    """
    main(cmdline=True)
