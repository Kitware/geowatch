import requests
import shutil
import os
from multiprocessing import Pool
from requests.exceptions import RequestException

from algorithm_toolkit import Algorithm, AlgorithmChain
from utils.globalmaptiles import GlobalMercator
from utils.image_utils import globalmercator_bounds_2_shapely_polygon

from shapely.wkt import loads


def get_tile(dat):
    tx, ty, zoom, output_dir = dat
    mercator = GlobalMercator()
    gx, gy = mercator.GoogleTile(tx, ty, zoom)

    # USGS limit is zoom level 16
    url_prefix = 'https://basemap.nationalmap.gov/ArcGIS/rest/services' + \
                 '/USGSImageryOnly/MapServer/tile/'
    url_coords = '%s/%s/%s' % (zoom, gy, gx)
    url = url_prefix + url_coords

    path = os.path.join(output_dir, '%s_%s_%s.png' % (gx, gy, zoom))

    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            if not os.path.exists(path):
                with open(path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                    return path
    except RequestException as e:
        # also catches ConnectionError, HttpError, etc.
        r = 'Connection failed on ' + url + '; error: ' + str(e)
        print(r)

    del r
    return None


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        self.logger.info("getting map tiles")

        output_dir = cl.get_temp_folder()

        roi_poly_wkt = params['roi']
        roi_poly = loads(roi_poly_wkt)
        bounds = roi_poly.bounds

        lat1 = bounds[1]
        lon1 = bounds[0]
        lat2 = bounds[3]
        lon2 = bounds[2]

        zoom = int(params['zoom'])

        mercator = GlobalMercator()
        mx, my = mercator.LatLonToMeters(lat1, lon1)
        tminx, tminy = mercator.MetersToTile(mx, my, zoom)

        mx, my = mercator.LatLonToMeters(lat2, lon2)
        tmaxx, tmaxy = mercator.MetersToTile(mx, my, zoom)

        txtytz = []
        for ty in range(tminy, tmaxy + 1):
            for tx in range(tminx, tmaxx + 1):
                tile_bounds = mercator.TileLatLonBounds(tx, ty, zoom)
                tile_poly = globalmercator_bounds_2_shapely_polygon(
                    tile_bounds)

                if tile_poly.intersects(roi_poly):
                    txtytz.append((tx, ty, zoom, output_dir))

        ntiles = len(txtytz)
        cl.set_status("starting to fetch " + str(ntiles) + " tiles", 0)

        p = Pool(8)
        filenames = []
        for i, path in enumerate(p.imap(get_tile, txtytz)):
            per = float(i) / float(ntiles) * 100.0
            if path is not None:
                filenames.append(path)
                cl.set_status('fetched tile: ' + path, round(per))

        # uncomment for no threading, can be slow
        # depending on connection speed!
        # [get_tile(dat) for dat in txtytz]

        cl.add_to_metadata('image_filenames', ",".join(filenames))

        return cl
