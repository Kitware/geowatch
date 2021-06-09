import os
import itertools
import json
import copy

from osgeo import gdal, osr

from algorithm_toolkit import Algorithm, AlgorithmChain
from watch.gis.spatial_reference import utm_epsg_from_latlon


def _aoi_bounds_to_utm_zone(aoi_bounds):
    '''
    Return majority EPSG code from aoi_bounds cornerpoints.  Returns
    first EPSG code if there's a tie
    '''
    lon0, lat0, lon1, lat1 = aoi_bounds

    codes = [utm_epsg_from_latlon(lat, lon)
             for lat, lon in itertools.product((lat0, lat1), (lon0, lon1))]

    code_counts = {}
    selected_code = None
    highest_count = 0
    for code in codes:
        code_counts[code] = code_counts.get(code, 0) + 1

        if code_counts[code] > highest_count:
            selected_code = code

    return selected_code


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        stac_catalog = copy.deepcopy(params['stac_catalog'])
        aoi_bounds = params['aoi_bounds']
        output_dir = params['output_dir']

        epsg_code = _aoi_bounds_to_utm_zone(aoi_bounds)

        for feature in stac_catalog.get('features', ()):
            try:
                # TODO: Process other asset types if present?
                input_path = feature['assets']['data']['href']
            except KeyError:
                continue

            input_base, input_ext = os.path.splitext(
                os.path.basename(input_path))

            feature_output_dir = os.path.join(output_dir, feature['id'])
            os.makedirs(feature_output_dir, exist_ok=True)

            output_path = os.path.join(
                feature_output_dir, "{}.tif".format(input_base))

            dst_crs = osr.SpatialReference()
            dst_crs.ImportFromEPSG(epsg_code)
            opts = gdal.WarpOptions(dstSRS=dst_crs, format="GTiff")
            out = gdal.Warp(output_path, input_path, options=opts)
            del out  # this is necessary, it writes out to disk

            feature['assets']['data']['href'] = output_path

        stac_catalog_output = {
            'output_type': 'text',
            'output_value': json.dumps(
                stac_catalog,
                indent=2, sort_keys=True)}

        output_dir_output = {
            'output_type': 'text',
            'output_value': params['output_dir']}

        cl.add_to_metadata('stac_catalog', stac_catalog_output)
        cl.add_to_metadata('output_dir', output_dir_output)

        # Do not edit below this line
        return cl
