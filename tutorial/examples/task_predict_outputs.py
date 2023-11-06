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
    # This demo will show how to add new assets / auxiliary images to a kwcoco
    # file.
    import watch
    import kwcoco
    import kwimage
    import numpy as np
    import ubelt as ub

    # Create a path where we can dumpy demo info
    demo_dpath = ub.Path.appdir('watch/demo/demo_add_auxiliary').ensuredir()

    # Start off by loading a demo dataset (this would be your dataset)
    dset = watch.coerce_kwcoco('watch-msi', geodata=True, dates=True)

    # We are going to add a new asset to each image.
    for image_id in dset.images():
        # Create a CocoImage object for each image.
        coco_image: kwcoco.CocoImage = dset.coco_image(image_id)

        image_name = coco_image.img['name']

        # When we add an asset we need to pass it the file path to that asset
        # and some additional information, namely the:
        # 1. filepath of the new asset
        # 2. the transform that warps the asset into "image space"
        # 3. Information about what channels / bands are in the image.
        # Other information can be added, but this tutorial currently only
        # covers the simple case.
        #coco_image.add_asset()

        # Let's pretend we've made a new asset for this coco image with 4 new
        # bands I will label as "foo" "bar" "baz" and "biz".

        # Let's create the demo image
        img_w = coco_image.img['width']
        img_h = coco_image.img['height']

        asset_w = img_w
        asset_h = img_h
        imdata = np.random.rand(asset_h, asset_w , 4)
        new_fpath = demo_dpath / f'image_{image_name}_myfeat.tif'
        kwimage.imwrite(new_fpath, imdata)

        # We will label it with this channel code:
        channels = 'foo|bar|baz|biz'

        # We need the transform that warps from asset space to image space
        # In this case they are aligned so we can just use the identity.
        warp_aux_to_img = kwimage.Affine.eye()

        # Use the CocoImage helper which will augment the coco dictionary with
        # your information.
        coco_image.add_asset(new_fpath, channels=channels, width=asset_w,
                             height=asset_h, warp_aux_to_img=warp_aux_to_img)
