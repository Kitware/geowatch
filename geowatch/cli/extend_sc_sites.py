import geowatch
import json
import geojson
import ubelt as ub
import dateutil
import datetime
from collections import defaultdict
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
        days = site['features'][1]['properties']['misc_info'][
            'phase_transition_days']
        days[0] += (sd - ssd).days
        obs0_feat = {}
        obs0_feat['geometry'] = geom
        obs0_feat['properties'] = default_feat()
        obs0_feat['properties']['current_phase'] = 'Site Preparation'
        obs0_feat['properties']['observation_date'] = start_date
        obs0_feat['properties']['misc_info'] = {
            'phase_transition_days': [days]
        }
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
        site['features'][-1]['properties']['misc_info'][
            'phase_transition_days'] = [days]
        obsn_feat = {}
        obsn_feat['geometry'] = site['features'][-1]['geometry']
        obsn_feat['properties'] = default_feat()
        obsn_feat['properties']['current_phase'] = 'Post Construction'
        obsn_feat['properties']['observation_date'] = send_date
        obsn_feat['properties']['misc_info'] = {'phase_transition_days': [1.0]}
        site['features'].append(obsn_feat)
        site['features'][0]['properties']['end_date'] = send_date

    return site


def extend_clone(bas_site, site):
    '''
    TODO does not handle phase_transition_days or enforce temporal consistency
    in phases.
    '''

    sstart_date = bas_site['features'][0]['properties']['start_date']
    send_date = bas_site['features'][0]['properties']['end_date']
    ssd, sed = to_date(sstart_date), to_date(send_date)
    start_date = site['features'][0]['properties']['start_date']
    end_date = site['features'][0]['properties']['end_date']
    sd, ed = to_date(start_date), to_date(end_date)
    site_id = bas_site['features'][0]['properties']['site_id']

    n_existing_feats = len(site['features']) - 1

    site_feats = defaultdict(list)
    for f in site['features'][1:]:
        k = to_date(f['properties']['observation_date'])
        site_feats[k].append(f)

    bas_site_feats = defaultdict(list)
    for f in bas_site['features'][1:]:
        k = to_date(f['properties']['observation_date'])
        bas_site_feats[k].append(f)

    if ssd < sd:
        print(f'extending start of {site_id} from {sd} to {ssd}')
        site['features'][0]['properties']['start_date'] = sstart_date

    if ed < sed:
        print(f'extending end of {site_id} from {ed} to {sed}')
        site['features'][0]['properties']['end_date'] = send_date

    for k in bas_site_feats:
        if k not in site_feats:
            site_feats[k] = bas_site_feats[k]

    site['features'] = (site['features'][:1] +
                        list(ub.flatten(ub.sorted_keys(site_feats).values())))

    n_after_feats = len(site['features']) - 1
    print(f'added {n_after_feats - n_existing_feats} features to {site_id}')

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
    obs0_feat['properties']['misc_info'] = {
        'phase_transition_days': [mid_days]
    }
    site.append(obs0_feat)

    obs1_feat = {}
    obs1_feat['geometry'] = geom
    obs1_feat['properties'] = default_feat()
    obs1_feat['properties']['current_phase'] = 'Active Construction'
    obs0_feat['properties']['observation_date'] = mid_date
    obs1_feat['properties']['misc_info'] = {
        'phase_transition_days': [end_days]
    }
    site.append(obs1_feat)

    obs2_feat = {}
    obs2_feat['geometry'] = geom
    obs2_feat['properties'] = default_feat()
    obs2_feat['properties']['current_phase'] = 'Post Construction'
    obs0_feat['properties']['observation_date'] = end_date
    obs2_feat['properties']['misc_info'] = {'phase_transition_days': [1.0]}
    site.append(obs2_feat)

    return {'type': 'FeatureCollection', 'features': site}


def main(region_models_pth,
         site_models_pth,
         out_region_models_pth,
         out_site_models_pth,
         modify_region_models_pth,
         modify_site_models_pth=None,
         create_missing=True,
         extend_existing=True):
    '''
    Modifies site models based on site summaries, and saves new site models and
    site summaries.

    Args:
        modify_site_models_pth: If given, create/extend from site models
        instead of site summaries
        create_missing: Create a new site model for any site summary that
        doesn't have one
        extend_existing: Extend the temporal duration of any site model smaller
        than its site summary

    Ignore:
        >>> from geowatch.cli.extend_sc_sites import main
        >>> main(
        >>>     'sc_out_region_models',
        >>>     'sc_out_site_models',
        >>>     'extended_sc_out_region_models',
        >>>     'extended_sc_out_site_models',
        >>>     'cropped_region_models_bas',
        >>> )
        >>> import jsonschema
        >>> import json
        >>> import geowatch
        >>> import ubelt as ub
        >>> REGION_SCHEMA = geowatch.rc.load_region_model_schema()
        >>> region_pths = ub.Path('extended_sc_out_region_models').glob('*.geojson')
        >>> regions = [json.load(open(p)) for p in region_pths]
        >>> for region in regions:
        >>>     # jsonschema.validate(region, schema=REGION_SCHEMA)
        >>>     pass
        >>> SITE_SCHEMA = geowatch.rc.load_site_model_schema()
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
    mr_pths = geowatch.utils.util_gis.coerce_geojson_paths(
        modify_region_models_pth)
    rm_pths = geowatch.utils.util_gis.coerce_geojson_paths(region_models_pth)
    ms_pths = geowatch.utils.util_gis.coerce_geojson_paths(modify_site_models_pth)
    sm_pths = geowatch.utils.util_gis.coerce_geojson_paths(site_models_pth)
    assert len(mr_pths) == 1, 'need 1 region -> n site models'
    assert len(rm_pths) == 1, 'need 1 region -> n site models'
    if len(sm_pths) < 1:
        print('warning: no existing site models found')

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

    msite_models = {}
    for pth in ms_pths:
        sm = json.load(open(pth))
        f = sm['features'][0]
        if (f['properties']['type'] == 'site'
                and f['properties']['model_content'] == 'proposed'):
            msite_models[f['properties']['site_id']] = sm

    site_models = {}
    for pth in sm_pths:
        sm = json.load(open(pth))
        f = sm['features'][0]
        if (f['properties']['type'] == 'site'
                and f['properties']['model_content'] == 'proposed'):
            site_models[f['properties']['site_id']] = sm

    default_summary_settings = next(iter(site_summaries.values())).copy()
    for k in ['start_date', 'end_date', 'site_id']:
        default_summary_settings['properties'].pop(k)
    rid = others[0]['properties']['region_id']

    if extend_existing:
        for k in site_models:
            if k in msite_models:
                sm = extend_clone(msite_models[k], site_models[k])
            else:
                sm = extend(msite_summaries[k], site_models[k])
            site_models[k] = sm
            ss = site_summaries[k]
            ss['start_date'] = sm['features'][0]['properties']['start_date']
            ss['end_date'] = sm['features'][0]['properties']['end_date']
            site_summaries[k] = ss

    if create_missing:
        for k in msite_summaries:
            if k not in site_models:
                if k in msite_models:
                    site_models[k] = msite_models[k]
                else:
                    sm = create(msite_summaries[k])
                ss = default_summary_settings.copy()
                ss['site_id'] = k
                ss['start_date'] = sm['features'][0]['properties'][
                    'start_date']
                ss['end_date'] = sm['features'][0]['properties']['end_date']
                site_models[k] = sm
                site_summaries[k] = ss

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
