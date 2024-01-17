# TA2 Feature Integration

The following is a set of high level instructions for integrating your feature model into the TA2 fusion system.

Recall that the input -- for both training and prediction -- into the TA2
fusion system is a kwcoco file. In the case of training the kwcoco file will
contain "images", "videos", and "annotations". In the case of prediction only
"images" and "videos" will be registered with the kwcoco file and it will be
the job of the fusion predictor to populate the "annotations" with labeled
(i.e. binary change/no-change for BAS, and fine-grained-activity label for
activity characterization) polygon predictions.

Thus, to provide your feature as input to the fusion module you must register
it with the kwcoco file. This is done by adding your rasterized feature map as
an auxiliary file for the image in question.

For instance consider the following kwcoco image json from drop1.

```python
{'id': 2,
 'file_name': None,
 'name': 'crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0',
 'auxiliary': [
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_coastal.tif',
       'channels': 'coastal',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 125,
       'width': 158,
       'warp_aux_to_img': {'scale': [6.0189873407872065, 6.008000000052702],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_blue.tif',
       'channels': 'blue',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 751,
       'width': 951,
       'warp_aux_to_img': {'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_green.tif',
       'channels': 'green',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 751,
       'width': 951,
       'warp_aux_to_img': {'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_red.tif',
       'channels': 'red',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 751,
       'width': 951,
       'warp_aux_to_img': {'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_B05.tif',
       'channels': 'B05',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 376,
       'width': 475,
       'warp_aux_to_img': {'scale': [2.0021052628305385, 1.9973404255478528],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_B06.tif',
       'channels': 'B06',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 376,
       'width': 475,
       'warp_aux_to_img': {'scale': [2.0021052628305385, 1.9973404255478528],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_B07.tif',
       'channels': 'B07',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 376,
       'width': 475,
       'warp_aux_to_img': {'scale': [2.0021052628305385, 1.9973404255478528],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_nir.tif',
       'channels': 'nir',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 751,
       'width': 951,
       'warp_aux_to_img': {'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_B09.tif',
       'channels': 'B09',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 125,
       'width': 158,
       'warp_aux_to_img': {'scale': [6.0189873407872065, 6.008000000052702],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_cirrus.tif',
       'channels': 'cirrus',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 125,
       'width': 158,
       'warp_aux_to_img': {'scale': [6.0189873407872065, 6.008000000052702],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_swir16.tif',
       'channels': 'swir16',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 376,
       'width': 475,
       'warp_aux_to_img': {'scale': [2.0021052628305385, 1.9973404255478528],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_swir22.tif',
       'channels': 'swir22',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 376,
       'width': 475,
       'warp_aux_to_img': {'scale': [2.0021052628305385, 1.9973404255478528],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_B8A.tif',
       'channels': 'B8A',
       'num_bands': 1,
       'sensor_coarse': 'S2',
       'height': 376,
       'width': 475,
       'warp_aux_to_img': {'scale': [2.0021052628305385, 1.9973404255478528],
        'type': 'affine'}},
      {'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0_r|g|b.tif',
       'channels': 'r|g|b',
       'num_bands': 3,
       'sensor_coarse': 'S2',
       'height': 751,
       'width': 951,
       'warp_aux_to_img': {'type': 'affine'}}],
 'height': 751,
 'width': 951,
 'utm_corners': [[433138.5566065939, 3359577.635952],
  [433138.5566065939, 3352066.8746835054],
  [442644.31407875166, 3352066.8746835054],
  [442644.31407875166, 3359577.635952]],
 'wld_crs_info': {'auth': ['EPSG', '32617'],
  'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER'},
 'utm_crs_info': {'auth': ['EPSG', '32617'],
  'axis_mapping': 'OAMS_AUTHORITY_COMPLIANT'},
 'wld_to_pxl': {'offset': [-43333.18712783946, 335923.71191231254],
  'scale': [0.10004463113912444, -0.09998986429647493],
  'type': 'affine'},
 'date_captured': '2018-10-16T16:02:29',
 'sensor_coarse': 'S2',
 'parent_file_name': None,
 'frame_index': 14,
 'timestamp': 736983,
 'video_id': 1,
 'warp_img_to_vid': {'scale': 1.01379425737959, 'type': 'affine'}}
```


This image currently registers the raw sensor bands. Thus the fusion module can access them.

Note that the base image has a "width" and "height" entry. This is the size of
the "canvas" that auxiliary images are warped onto when sampling via ndsampler
or with kwcoco's new `delayed_load` feature. 

Also note that each auxiliary dictionary has a "width" and "height", which are
the size of the actual images on disk. The `warp_aux_to_img` field indicates
how to warp the native auxiliary space onto the image canvas. Likewise the
image has a `warp_img_to_vid` attribute which is used to align the images in a
sequence, but that is not relevant when we are operating on a per-image basis.

To add your band you would add another entry to the image's "auxiliary" list.

For instance, say you were adding "invariant" features (with 8 channels)
computed by your neural network. In your prediction script, you would read in
the input kwcoco, and then write an augmented output kwcoco file, where the new
entry in each image's auxiliary list might look like this:

```python
{
    'file_name': 'US_Jacksonville_R01/S2/affine_warp/crop_2018-10-16_N30.298499W081.695322_N30.366761W081.596884_S2_0/DATA_FOR_YOUR_NEW_FEATURE.tif',
    'channels': 'myinv1|myinv2|myinv3|myinv4|myinv5|myinv6|myinv7|myinv8',
    'num_bands': 8,
    'height': 751,
    'width': 951,
    'warp_aux_to_img': {'scale': [1.0, 1.0], 'type': 'affine'
}
```

Now the fusion module will be able to fuse in your feature by referencing the
`myinv1|myinv2|myinv3|myinv4|myinv5|myinv6|myinv7|myinv8` channel code. Note
that in this example I gave a name to each individual band in the set of
invariant features delimited by a `|`. The current state of the code may be
sensitive to this, but the plan is that this could be a more concise alias and
the `num_bands` item will specify how many channels are actually stored in the
feature to be fused.


In terms of actionable items, we would like each subteam producing a TA2
feature to add their features to the datasets on DVC. While doing this each
subteam should ensure they have a "predict.py" script as previously discussed
that inputs and outputs kwcoco so each subteams features can be computed for
new data.

The datasets we would like to immediately target are:

https://gitlab.kitware.com/smart/smart_watch_dvc/-/tree/master/drop1_S2_aligned_c1

and 

https://gitlab.kitware.com/smart/smart_watch_dvc/-/tree/master/drop1-S2-L8-LS-aligned-c1


NOTES:

When you write your raster features to disk it will be important to write them
as a cloud optimized geotiff. This can be done with kwimage:

```python
    # image_path: will be the file you are writing your auxiliary data to
    # image_data: is a H x W x C array where C is the number of channels
    # space=None tells imwrite that you are not saving RGB, so it wont complain about multiple channels
    # using backend='gdal' will write the data as a tiled GeoTiff
    kwimage.imwrite(image_path, image_data, space=None, backend='gdal')
```



----

Notes for slides

Input KWCOCO File:


```
{
"videos": [
    {"name": "TheRegionName", "width": 300, "height": 400},
 ...],
"images": [
    { 
        "name": "TheImageName",
        "width": 600,
        "height": 800,
        "video_id": 1,
        "date_captured": "2018-10-16T16:02:29",
        "warp_img_to_vid": {"scale": 0.5},
        "auxiliary": [
            {
                "file_name": "B1.tif", 
                "warp_aux_to_img": {"scale": 2.0}, 
                "width": 300, "height": 400
                "channels": "coastal", "num_bands": 1,
            },
            { 
                "file_name": "B2.tif", 
                "warp_aux_to_img": {"scale": 1.0},
                "channels": "blue", "num_bands": 1,
            },
            ...
        ],
    }, ...  ]
}
```


Output KWCOCO File, Simply append to the "auxiliary list" in each appropriate
image dictionary.

```
...
"auxiliary": [
    {"file_name": "B1.tif", ...},
    {"file_name": "B2.tif", ...},
    { 
        "file_name": "YOUR_FEATURE_PATH.tif", 
        "warp_aux_to_img": {"scale": 4.0},
        "width": 75, "height": 100,
        "channels": "your_channel_code", 
        "num_bands": 32,
        ...
    },
    ...
]
```
