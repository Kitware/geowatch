"""
This describes the structure of how a task prediction script should work.


What must happen is that there must be a predict.py script that accepts a
kwcoco as input and produces a kwcoco as output. The output kwcoco should have
additional "auxiliary" or "asset" images added to each "coco image". These new
assets are rasters that can contain any number of channels.

These are the features that will be used as input to the fusion network.

The existing invariants/predict.py code (as well as fusion/predict.py and every
other task/*/predict.py) has logic to run a model over a sliding window and
stitch the rasters back together such that they align to the larger image they
belong to (up to an affine transformation).


For example an input coco image dictionary may look like this:

{
    'date_captured': '1984-05-24T14:44:38.034876',
    'file_name': None,
    'frame_index': 0,
    'height': 661,
    'width': 778
    'id': 1,
    'name': 'generated-1-0',
    'sensor': 'sensor1',
    'sensor_coarse': 'sensor0',
    'valid_region': [{
        'exterior':
            [[0.0000, 0.0000], [0.0000, 661.0000], [778.0000, 661.0000],
            [778.0000, 0.0000], [0.0000, 0.0000]
        ],
        'interiors': []}],
    'valid_region_utm': {
        'coordinates': [
        [[[496408.5193, -5231846.5362], [496408.5193, -5231712.3362], [496530.5193, -5231712.3362],
        [496530.5193, -5231846.5362], [496408.5193, -5231846.5362]]]],
        'properties': {
            'crs': {'auth': ['EPSG', '32620'], 'axis_mapping': 'OAMS_AUTHORITY_COMPLIANT'}
        },
        'type': 'MultiPolygon'
    },
    'geos_corners': {
        'coordinates': [
            [[-63.0475, -47.2401], [-63.0475, -47.2389],
                [-63.0458, -47.2389], [-63.0458, -47.2401]] ],
        'properties': {
            'crs_info': {'auth': ['EPSG', '4326'], 'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER'}
        },
        'type': 'Polygon'
    },
    'video_id': 1,
    'warp_img_to_vid': {
        'offset': [0.0000, 0.0000],
        'scale': [0.7236, 0.5203],
        'shearx': -0.0000,
        'theta': -0.0000,
        'type': 'affine'
    },

    'auxiliary': [
        {
            'approx_meter_gsd': 0.1792,
            'band_metas': [{'nodata': None}],
            'height': 661,
            'width': 778,
            'num_bands': 1,
            'is_rpc': False,
            'channels': 'B1',
            'file_name': '_assets/aux/aux_B1/img_00001.tif',
            'warp_aux_to_img': {'type': 'affine', 'scale': 1},
        }

        {
            'approx_meter_gsd': 0.5638,
            'band_metas': [{'nodata': None}],
            'channels': 'B8',
            'file_name': '_assets/aux/aux_B8/img_00001.tif',
            'height': 111,
            'is_rpc': False,
            'num_bands': 1,
            'warp_aux_to_img': {'scale': 6.0000, 'type': 'affine'},
            'width': 130,
        },

    ]
}

"""


def demo():
    from watch.demo import coerce_kwcoco
    dset = coerce_kwcoco(geodata=True, dates=True)
