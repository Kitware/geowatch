"""
Script to find landsat images that have interesting overlap for
demodata for the align script.
"""

def main():
    import kwimage
    import dateutil
    import json
    import fels
    import datetime
    import os

    cats_path = os.path.expanduser('~/smart/data/fels/')
    geojson_box = kwimage.Boxes.from_slice(
        (slice(54.8, 55.2), slice(23.8, 24.4))).to_polygons()[0].to_geojson()
    pad = datetime.timedelta(days=0.5)
    dt_min = dateutil.parser.parse('2019-10-20') - pad
    dt_max = dateutil.parser.parse('2019-10-20') + pad
    geometry = json.dumps(geojson_box)
    sat = 'OLI_TIRS'
    scenes = fels.fels.convert_wkt_to_scene(
        sat='OLI_TIRS',  # l8
        geometry=geometry,
        include_overlap=True
    )
    landsat_metadata_file = fels.fels.ensure_landsat_metadata()

    cc_values = []
    all_urls = []
    all_acqdates = []

    for scene in scenes:
        print('scene = {!r}'.format(scene))
        wr2path = scene[0:3]
        wr2row = scene[3:6]
        cc_limit = 100
        date_start = dt_min.date().isoformat()
        date_end = dt_max.date().isoformat()
        url = fels.landsat.query_landsat_catalogue(
            landsat_metadata_file,
            cc_limit=100,
            date_start=dt_min.date().isoformat(),
            date_end=dt_max.date().isoformat(),
            wr2path=scene[0:3],
            wr2row=scene[3:6],
            sensor=sat,
            latest=False,
            use_csv=False
        )
        print('url = {!r}'.format(url))

        sensor = sat
        conn = fels.landsat._ensure_landsat_sqlite_conn(landsat_metadata_file)
        cur = conn.cursor()

        try:
            result = cur.execute(
                '''
                SELECT BASE_URL, CLOUD_COVER, DATE_ACQUIRED from landsat WHERE

                WRS_PATH=? AND WRS_ROW=? AND SENSOR_ID=? AND CLOUD_COVER <= ?
                ''', (
                    int(wr2path),
                    int(wr2row),
                    sensor,
                    cc_limit
                ))
            for found in result:
                all_urls.append(found[0])
                cc_values.append(found[1])
                all_acqdates.append(dateutil.parser.isoparse(found[2]))
        finally:
            cur.close()


