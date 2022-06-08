

def demo_khq_region_data():
    """
    A small demo region around KHQ while it is being built
    """
    import mgrs
    mgrs_code = mgrs.MGRS().toMGRS(42.865978051904726, 73.77529621124268, MGRSPrecision=0)

    khq_region_geom = {
        "type": "Polygon",
        "coordinates": [
            [[-73.77379417419434, 42.86254939745846],
             [-73.76715302467346, 42.86361104246733],
             [-73.76901984214783, 42.86713400027327],
             [ -73.77529621124268, 42.865978051904726],
             [ -73.7755537033081, 42.86542759269259],
             [ -73.7750494480133, 42.862525805139775],
             [ -73.77379417419434, 42.86254939745846]]
        ]
    }

    khq_sitesum_geom = {
        "type": "Polygon",
        "coordinates": [
            [[-73.77200379967688, 42.864783745778894],
             [-73.77177715301514, 42.86412514733195],
             [-73.77110660076141, 42.8641654498268],
             [-73.77105563879013, 42.86423720786224],
             [-73.7710489332676, 42.864399400374786],
             [-73.77134531736374, 42.8649134986743],
             [-73.77200379967688, 42.864783745778894]]
        ]
    }

    khq_region_feature = {
        "type": "Feature",
        "properties": {
            "type": "region",
            "region_id": "KHQ_R001",
            "version": "2.4.3",
            "mgrs": mgrs_code,
            "start_date": "2017-01-01",
            "end_date": "2020-01-01",
            "originator": "kit-demo",
            "model_content": "annotation",
            "comments": None,
        },
        "geometry": khq_region_geom
    }

    khq_sitesum_feature = {
        "type": "Feature",
        "properties": {
            "type": "site_summary",
            "status": "positive_annotated",
            "version": "2.0.0",
            "site_id": "KHQ_R001_0000",
            "mgrs": mgrs_code,
            "start_date": "2018-01-01",
            "end_date": "2019-01-01",
            "score": 1.0,
            "originator": "kit-demo",
            "model_content": "annotation",
            "validated": "False"
        },
        'geometry': khq_sitesum_geom,
    }

    region_data = {
        'type': 'FeatureCollection',
        'features': [
            khq_region_feature,
            khq_sitesum_feature,
        ]
    }
    return region_data


def demo_khq_region_fpath():
    import json
    import ubelt as ub
    data = demo_khq_region_data()
    dpath = ub.Path.appdir('watch/demo/regions').ensuredir()
    fpath = dpath / 'KHQ_R001.geojson'
    fpath.write_text(json.dumps(data))
    return fpath
