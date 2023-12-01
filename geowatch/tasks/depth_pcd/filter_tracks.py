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
    from geowatch.geoannots import geomodels
    from geowatch.utils import util_gis
    from kwcoco.util import util_json
    from geowatch.utils import process_context

    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(config.asdict())
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    proc_context = process_context.ProcessContext(
        name='geowatch.tasks.depth_pcd.filter_tracks', type='process',
        config=jsonified_config,
        track_emissions=False,
    )
    proc_context.start()

    # Read the region / site models we will modify
    region_model = geomodels.RegionModel.coerce(config.input_region)
    input_site_fpaths = util_gis.coerce_geojson_paths(config.input_sites)
    site_to_site_fpath = ub.udict({
        p.stem: p for p in input_site_fpaths
    })
    site_id_to_summary = ub.udict()
    for summary in region_model.site_summaries():
        assert summary.site_id not in site_id_to_summary
        site_id_to_summary[summary.site_id] = summary
    input_sites = list(geomodels.SiteModel.coerce_multiple(site_to_site_fpath.values()))

    if __debug__:
        # Check that the site paths correspond with the input site summary.
        # If they don't the following logic will produce unexpected results.
        sites_with_paths = set(site_to_site_fpath)
        sites_with_summary = set(site_id_to_summary)
        if sites_with_paths != sites_with_summary:
            print('')
            print('sites_with_paths = {}'.format(ub.urepr(sites_with_paths, nl=1)))
            print('sites_with_summary = {}'.format(ub.urepr(sites_with_summary, nl=1)))
            raise AssertionError(
                f'sites with paths {len(sites_with_paths)} are not the same as '
                f'sites with summaries {len(sites_with_summary)}')

    # Ensure caches
    for summary in site_id_to_summary.values():
        summary['properties'].setdefault('cache', {})
    for site in input_sites:
        site.header['properties'].setdefault('cache', {})

    # Build a decision and reasons for each site
    site_to_decisions = {
        s: {
            'type': 'depth_decision',
            'accept': True,
            'why': None,
            'score': None,
        }
        for s in site_id_to_summary.keys()
    }

    # track_ids_to_drop = []
    coco_dset = kwcoco.CocoDataset.coerce(config.input_kwcoco)

    # TODO: use new kwcoco track mechanisms
    tracks = coco_dset.dataset['tracks']
    tracks = [t for t in tracks if t['src'] == 'sv_depth_pcd']

    for t in tracks:
        # We are assuming track-ids correspond to site names here.
        site_id = t['id']
        if isinstance(site_id, int):
            raise NotImplementedError('new kwcoco track mechanisms')
        decision = site_to_decisions[site_id]
        accept = (t['score'] >= config.threshold)
        decision['accept'] = bool(accept)
        decision['score'] = float(t['score'])
        decision['why'] = 'threshold'

    num_accept = sum((d['accept']) for d in site_to_decisions.values())
    print(f'Filter to {num_accept} / {len(site_id_to_summary)} sites')

    # Enrich each site summary with the decision reason and update status
    for site_id, decision in site_to_decisions.items():
        sitesum = site_id_to_summary[site_id]
        # Change the status of sites to "system_rejected" instead of droping them
        if not decision['accept']:
            sitesum['properties']['status'] = 'system_rejected'
        sitesum['properties']['cache']['depth_decision'] = decision

    # Copy the site models and update their header with new summary
    # information.
    output_sites_dpath = ub.Path(config.output_sites_dpath)
    output_sites_dpath.ensuredir()
    out_site_fpaths = []

    for old_site in input_sites:
        old_fpath = site_to_site_fpath[old_site.site_id]
        new_fpath = output_sites_dpath / old_fpath.name
        new_summary = site_id_to_summary[site_id]
        old_site.header['properties']['status'] = new_summary['properties']['status']
        old_site.header['properties']['cache'].update(new_summary['properties']['cache'])
        new_fpath.write_text(old_site.dumps())
        out_site_fpaths.append(new_fpath)

    # Write the updated site summaries in a new region model
    new_summaries = list(site_id_to_summary.values())
    new_region_model = geomodels.RegionModel.from_features(
        [region_model.header] + list(new_summaries))
    output_region_fpath = ub.Path(config.output_region_fpath)
    output_region_fpath.parent.ensuredir()
    print(f'Write filtered region model to: {output_region_fpath}')
    with safer.open(output_region_fpath, 'w', temp_file=not ub.WIN32) as file:
        json.dump(new_region_model, file, indent=4)

    proc_context.stop()

    if config.output_site_manifest_fpath is not None:
        filter_output = {'type': 'tracking_result', 'info': [], 'files': [os.fspath(p) for p in out_site_fpaths]}
        filter_output['info'].append(proc_context.obj)
        print(f'Write filtered site result to {config.output_site_manifest_fpath}')
        with safer.open(config.output_site_manifest_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(filter_output, file, indent=4)


if __name__ == '__main__':
    main()
