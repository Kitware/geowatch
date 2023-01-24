import watch
import json
import geojson
import ubelt as ub
import dateutil
import datetime
# import pandas as pd
# import geopandas as gpd


def default_feat():
    return {
        'is_occluded': 'False',
        'is_site_boundary': 'True',
        'type': 'observation',
        'source': 'DUMMY',
        'sensor_name': 'Sentinel-2',
        'score': 1.0
    }


def to_date(date_str):
    return dateutil.parser.parse(date_str).date()


# TODO should phase_transition_days be in a list []? It is in the delivered
# system.
def extend(site_summary, site):

    sstart_date = site_summary['properties']['start_date']
    send_date = site_summary['properties']['end_date']
    ssd, sed = to_date(sstart_date), to_date(send_date)
    start_date = site['features'][0]['properties']['start_date']
    end_date = site['features'][0]['properties']['end_date']
    sd, ed = to_date(start_date), to_date(end_date)
    site_id = site_summary['properties']['site_id']

    if ssd < sd:
        print(f'extending start of {site_id} from {sd} to {ssd}')
        geom = site['features'][1]['geometry']
        days = site['features'][1]['properties']['misc_info']['phase_transition_days']
        days[0] += (sd - ssd).days
        obs0_feat = {}
        obs0_feat['geometry'] = geom
        obs0_feat['properties'] = default_feat()
        obs0_feat['properties']['current_phase'] = 'Site Preparation'
        obs0_feat['properties']['observation_date'] = start_date
        obs0_feat['properties']['misc_info'] = {'phase_transition_days': [days]}
        site['features'].insert(1, obs0_feat)
        site['features'][0]['properties']['start_date'] = sstart_date

    if ed < sed:
        print(f'extending end of {site_id} from {ed} to {sed}')
        try:
            phase = site['features'][-2]['properties']
        except IndexError:
            phase = 'Active Construction'
        site['features'][-1]['properties']['current_phase'] = phase
        if phase == 'Post Construction':
            days = 1.0
        else:
            days = (sed - ed).days
        site['features'][-1]['properties']['misc_info']['phase_transition_days'] = [days]
        obsn_feat = {}
        obsn_feat['geometry'] = site['features'][-1]['geometry']
        obsn_feat['properties'] = default_feat()
        obsn_feat['properties']['current_phase'] = 'Post Construction'
        obsn_feat['properties']['observation_date'] = send_date
        obsn_feat['properties']['misc_info'] = {'phase_transition_days': [1.0]}
        site['features'].append(obsn_feat)

    return site


def create(site_summary):
    geom = site_summary['geometry']
    start_date = site_summary['properties']['start_date']
    end_date = site_summary['properties']['end_date']
    site_id = site_summary['properties']['site_id']

    sd, ed = to_date(start_date), to_date(end_date)
    tot_days = (ed - sd).days
    mid_days = int(tot_days / 2)
    end_days = tot_days - mid_days
    mid_date = (sd + datetime.timedelta(days=mid_days)).isoformat()
    print(f'creating {site_id} from {sd} to {ed}')

    site = []

    site_feat = site_summary.copy()
    site_feat['properties']['type'] = 'site'
    site_feat['properties']['region_id'] = '_'.join(site_id.split('_')[:-1])
    site_feat['properties']['misc_info'] = {}
    site.append(site_feat)

    obs0_feat = {}
    obs0_feat['geometry'] = geom
    obs0_feat['properties'] = default_feat()
    obs0_feat['properties']['current_phase'] = 'Site Preparation'
    obs0_feat['properties']['observation_date'] = start_date
    obs0_feat['properties']['misc_info'] = {'phase_transition_days': [mid_days]}
    site.append(obs0_feat)

    obs1_feat = {}
    obs1_feat['geometry'] = geom
    obs1_feat['properties'] = default_feat()
    obs1_feat['properties']['current_phase'] = 'Active Construction'
    obs0_feat['properties']['observation_date'] = mid_date
    obs1_feat['properties']['misc_info'] = {'phase_transition_days': [end_days]}
    site.append(obs1_feat)

    obs2_feat = {}
    obs2_feat['geometry'] = geom
    obs2_feat['properties'] = default_feat()
    obs2_feat['properties']['current_phase'] = 'Post Construction'
    obs0_feat['properties']['observation_date'] = end_date
    obs2_feat['properties']['misc_info'] = {'phase_transition_days': [1.0]}
    site.append(obs2_feat)

    return {'type': 'FeatureCollection', 'features': site}


def main(modify_region_models_pth,
         region_models_pth,
         site_models_pth,
         out_region_models_pth,
         out_site_models_pth,
         create_missing=True,
         extend_existing=True):
    '''
    Modifies site models based on site summaries, and saves new site models and
    site summaries.

    Args:
        create_missing: Create a new site model for any site summary that
        doesn't have one
        extend_existing: Extend the temporal duration of any site model smaller
        than its site summary

    Ignore:
        >>> from watch.cli.extend_sc_sites import main
        >>> main(
        >>>     'cropped_region_models_bas',
        >>>     'sc_out_region_models',
        >>>     'sc_out_site_models',
        >>>     'extended_sc_out_region_models',
        >>>     'extended_sc_out_site_models'
        >>> )
        >>> import jsonschema
        >>> import json
        >>> import watch
        >>> import ubelt as ub
        >>> REGION_SCHEMA = watch.rc.load_region_model_schema()
        >>> region_pths = ub.Path('extended_sc_out_region_models').glob('*.geojson')
        >>> regions = [json.load(open(p)) for p in region_pths]
        >>> for region in regions:
        >>>     # jsonschema.validate(region, schema=REGION_SCHEMA)
        >>>     pass
        >>> SITE_SCHEMA = watch.rc.load_site_model_schema()
        >>> site_pths = ub.Path('extended_sc_out_site_models').glob('*.geojson')
        >>> sites = [json.load(open(p)) for p in site_pths]
        >>> for site in sites:
        >>>     # jsonschema.validate(site, schema=SITE_SCHEMA)
        >>>     pass
        >>> # new way to validate sites
        >>> from iarpa_smart_metrics.evaluation import SiteStack
        >>> for p in site_pths:
        >>>     s = SiteStack(p)

    '''
    mr_pths = watch.utils.util_gis.coerce_geojson_paths(
        modify_region_models_pth)
    rm_pths = watch.utils.util_gis.coerce_geojson_paths(region_models_pth)
    sm_pths = watch.utils.util_gis.coerce_geojson_paths(site_models_pth)
    assert len(mr_pths) == 1, 'need 1 region -> n site models'
    assert len(rm_pths) == 1, 'need 1 region -> n site models'
    if len(sm_pths) < 1:
        print('warning: no existing site models found')
    '''
    regions = pd.concat([
        i['data'] for i in watch.utils.util_gis.coerce_geojson_datas(rm_pths)
    ])

    site_summaries = regions[(regions['type'] == 'site_summary')
                             & (regions['model_content'] == 'proposed')]
    site_summaries = site_summaries.set_index(['region_id', 'site_id', 'type'],
                                              drop=False,
                                              append=True)

    sites = pd.concat([
        i['data'] for i in watch.utils.util_gis.coerce_geojson_datas(sm_pths)
    ])
    # sites = sites[(sites['type'] == 'site')
                  # & (sites['model_content'] == 'proposed')]
    sites = sites.set_index(['region_id', 'site_id', 'type'],
                            drop=False,
                            append=True)
    '''
    msite_summaries = {}
    for pth in mr_pths:
        rm = json.load(open(pth))
        for f in rm['features']:
            if (f['properties']['type'] == 'site_summary'
                    and f['properties']['model_content'] == 'proposed'):
                msite_summaries[f['properties']['site_id']] = f

    site_summaries = {}
    others = []
    for pth in rm_pths:
        rm = json.load(open(pth))
        for f in rm['features']:
            if (f['properties']['type'] == 'site_summary'
                    and f['properties']['model_content'] == 'proposed'):
                site_summaries[f['properties']['site_id']] = f
            else:
                others.append(f)

    site_models = {}
    for pth in sm_pths:
        sm = json.load(open(pth))
        f = sm['features'][0]
        if (f['properties']['type'] == 'site' and f['properties']['model_content'] == 'proposed'):
            site_models[f['properties']['site_id']] = sm

    default_summary_settings = next(iter(site_summaries.values())).copy()
    for k in ['start_date', 'end_date', 'site_id']:
        default_summary_settings['properties'].pop(k)
    rid = others[0]['properties']['region_id']
    # next_si = max(int(f['properties']['site_id'].split('_')[-1]) for f in msite_summaries.values()) + 1

    if extend_existing:
        for k in site_models:
            sm = extend(msite_summaries[k], site_models[k])
            site_models[k] = sm
            ss = site_summaries[k]
            ss['start_date'] = sm['features'][0]['properties']['start_date']
            ss['end_date'] = sm['features'][0]['properties']['end_date']
            site_summaries[k] = ss

    if create_missing:
        for k in msite_summaries:
            if k not in site_models:
                sm = create(
                    msite_summaries[k],
                    # '_'.join((rid, str(next_si).zfill(4)))
                )
                ss = default_summary_settings.copy()
                ss['site_id'] = k
                ss['start_date'] = sm['features'][0]['properties']['start_date']
                ss['end_date'] = sm['features'][0]['properties']['end_date']
                site_models[k] = sm
                site_summaries[k] = ss
                # next_si += 1

    new_rm = geojson.FeatureCollection(others + list(site_summaries.values()))
    with open(
            ub.Path(out_region_models_pth).ensuredir() / f'{rid}.geojson',
            'w') as f:
        json.dump(new_rm, f, indent=2)

    for sm in site_models.values():
        sid = sm['features'][0]['properties']['site_id']
        with open(
                ub.Path(out_site_models_pth).ensuredir() / f'{sid}.geojson',
                'w') as f:
            json.dump(sm, f, indent=2)
