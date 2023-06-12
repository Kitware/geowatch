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


def main(cmdline=1, **kwargs):
    config = FilterTracksConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich

    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    import kwcoco
    import safer
    from watch.geoannots import geomodels
    from watch.utils import util_gis
    import pandas as pd
    from kwcoco.util import util_json
    from watch.utils import process_context

    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(config.asdict())
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    proc_context = process_context.ProcessContext(
        name='watch.tasks.depth_pcd.filter_tracks', type='process',
        config=jsonified_config,
        track_emissions=False,
    )
    proc_context.start()

    track_ids_to_drop = []
    coco_dset = kwcoco.CocoDataset.coerce(config.input_kwcoco)
    tracks = pd.DataFrame(coco_dset.dataset['tracks'])
    tracks = tracks[tracks['src'] == 'sv_depth_pcd'].values.tolist()

    for t in tracks:
        if t[2] < config.threshold:
            track_ids_to_drop.append(t[0])

    print(f"Dropping {len(track_ids_to_drop)} / {len(tracks)} tracks")

    region_model = geomodels.RegionModel.coerce(config.input_region)
    input_site_fpaths = util_gis.coerce_geojson_paths(config.input_sites)
    site_to_site_fpath = ub.udict({
        p.stem: p for p in input_site_fpaths
    })
    site_id_to_summary = ub.udict()
    for summary in region_model.site_summaries():
        assert summary.site_id not in site_id_to_summary
        site_id_to_summary[summary.site_id] = summary
    # output_region_fpath = ub.Path(config.output_region_fpath)

    # We are assuming track-ids correspond to site names here.
    assert set(site_id_to_summary).issuperset(track_ids_to_drop)

    keep_summaries = ub.udict(site_id_to_summary) - track_ids_to_drop
    keep_site_fpaths = ub.udict(site_to_site_fpath) - track_ids_to_drop
    reject_sites = set(site_to_site_fpath) - set(keep_site_fpaths)

    if __debug__:
        # Check that the site paths correspond with the input site summary.
        # If they don't the following logic will produce unexpected results.
        sites_with_paths = set(keep_summaries)
        sites_with_summary = set(keep_site_fpaths)
        if sites_with_paths != sites_with_summary:
            print('sites_with_paths = {}'.format(ub.urepr(sites_with_paths, nl=1)))
            print('sites_with_summary = {}'.format(ub.urepr(sites_with_summary, nl=1)))
            raise AssertionError(
                f'sites with paths {len(sites_with_paths)} are not the same as '
                f'sites with summaries {len(sites_with_summary)}')

    new_summaries = list(keep_summaries.values())

    MARK_INSTEAD_OF_REMOVE = 1
    if MARK_INSTEAD_OF_REMOVE:
        # Change the status of sites to "system_rejected" instead of droping
        # them
        reject_summaries = list(site_id_to_summary.subdict(reject_sites).values())
        for sitesum in reject_summaries:
            sitesum['properties']['status'] = 'system_rejected'
        new_summaries.extend(reject_summaries)

    # Copy the filtered site models over to the output directory
    output_sites_dpath = ub.Path(config.output_sites_dpath)
    output_sites_dpath.ensuredir()

    out_site_fpaths = []
    # Copy accepted sites without any modification
    for old_fpath in keep_site_fpaths.values():
        new_fpath = output_sites_dpath / old_fpath.name
        old_fpath.copy(new_fpath, overwrite=True)
        out_site_fpaths.append(new_fpath)

    if MARK_INSTEAD_OF_REMOVE:
        reject_site_fpaths = site_to_site_fpath.subdict(reject_sites)
        # Copy the rejected sites as well, but modify their status
        for old_fpath in reject_site_fpaths.values():
            new_fpath = output_sites_dpath / old_fpath.name
            old_site = geomodels.SiteModel.coerce(old_fpath)
            old_site.header['properties']['status'] = 'system_rejected'
            new_fpath.write_text(old_site.dumps())
            out_site_fpaths.append(new_fpath)

    new_region_model = geomodels.RegionModel.from_features(
        [region_model.header] + new_summaries)

    output_region_fpath = ub.Path(config.output_region_fpath)
    output_region_fpath.parent.ensuredir()

    with safer.open(output_region_fpath, 'w', temp_file=not ub.WIN32) as file:
        json.dump(new_region_model, file, indent=4)

    if config.output_site_manifest_fpath is not None:
        filter_output = {'type': 'tracking_result', 'info': [], 'files': [os.fspath(p) for p in out_site_fpaths]}
        filter_output['info'].append(proc_context.obj)
        print(f'Write filtered site result to {config.output_site_manifest_fpath}')
        with safer.open(config.output_site_manifest_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(filter_output, file, indent=4)


if __name__ == '__main__':
    main()
