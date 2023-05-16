import json
import os
import ubelt as ub
import scriptconfig as scfg


class FilterTracksConfig(scfg.DataConfig):
    input_kwcoco = scfg.Value(None, required=True, help=ub.paragraph(
        '''
        The input kwcoco file with the scores.
        This does not to cover all sites, any site this does not
        cover will be automatically accepted.
        '''), position=1, alias=['in_file'], group='inputs')

    threshold = scfg.Value(0.4, help=ub.paragraph(
        '''
            threshold to filter polygons, very sensitive
            '''), group='track scoring')

    input_region = scfg.Value(None, help='The coercable input region model', group='inputs')

    input_sites = scfg.Value(None, help='The coercable input site models', group='inputs')

    output_region_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The output for the region with filtered site summaries
        '''), group='outputs')

    output_sites_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        The directory where site model geojson files will be written.
        '''), alias=['out_sites_dir'], group='outputs')

    output_site_manifest_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site models will be written.
        '''), alias=['out_sites_fpath'], group='outputs')


def main(**kwargs):
    args = FilterTracksConfig.cli(cmdline=True, data=kwargs, strict=True)
    import rich

    rich.print('args = {}'.format(ub.urepr(args, nl=1)))

    import kwcoco
    import safer
    from watch.geoannots import geomodels
    from watch.utils import util_gis
    import pandas as pd
    track_ids_to_drop = []

    coco_dset = kwcoco.CocoDataset.coerce(args.input_kwcoco)
    tracks = pd.DataFrame(coco_dset.dataset['tracks'])
    tracks = tracks[tracks['src'] == 'sv_depth_pcd'].values.tolist()

    for t in tracks:
        if t[2] < args.threshold:
            track_ids_to_drop.append(t[0])

    print(f"Dropping {len(track_ids_to_drop)} / {len(tracks)} tracks")

    region_model = geomodels.RegionModel.coerce(args.input_region)
    input_site_fpaths = util_gis.coerce_geojson_paths(args.input_sites)
    site_to_site_fpath = ub.udict({
        p.stem: p for p in input_site_fpaths
    })
    site_id_to_summary = {}
    for summary in region_model.site_summaries():
        assert summary.site_id not in site_id_to_summary
        site_id_to_summary[summary.site_id] = summary
    # output_region_fpath = ub.Path(args.output_region_fpath)

    # We are assuming track-ids correspond to site names here.
    assert set(site_id_to_summary).issuperset(track_ids_to_drop)

    keep_summaries = ub.udict(site_id_to_summary) - track_ids_to_drop
    keep_site_fpaths = ub.udict(site_to_site_fpath) - track_ids_to_drop

    sites_with_paths = set(keep_summaries)
    sites_with_summary = set(keep_site_fpaths)
    if sites_with_paths != sites_with_summary:
        print('sites_with_paths = {}'.format(ub.urepr(sites_with_paths, nl=1)))
        print('sites_with_summary = {}'.format(ub.urepr(sites_with_summary, nl=1)))
        raise AssertionError(
            f'sites with paths {len(sites_with_paths)} are not the same as '
            f'sites with summaries {len(sites_with_summary)}')

    # Copy the filtered site models over to the output directory
    output_sites_dpath = ub.Path(args.output_sites_dpath)
    output_sites_dpath.ensuredir()
    out_site_fpaths = []
    for old_fpath in keep_site_fpaths.values():
        new_fpath = output_sites_dpath / old_fpath.name
        old_fpath.copy(new_fpath, overwrite=True)
        out_site_fpaths.append(new_fpath)

    new_region_model = geomodels.RegionModel.from_features(
        [region_model.header] + list(keep_summaries.values()))

    output_region_fpath = ub.Path(args.output_region_fpath)
    output_region_fpath.parent.ensuredir()

    with safer.open(output_region_fpath, 'w', temp_file=not ub.WIN32) as file:
        json.dump(new_region_model, file, indent=4)

    if args.output_site_manifest_fpath is not None:
        filter_output = {'type': 'tracking_result', 'info': [], 'files': [os.fspath(p) for p in out_site_fpaths]}
        # filter_output['info'].append(proc_context.obj)
        print(f'Write filtered site result to {args.output_site_manifest_fpath}')
        with safer.open(args.output_site_manifest_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(filter_output, file, indent=4)


if __name__ == '__main__':
    main()
