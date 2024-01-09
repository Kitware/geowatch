"""
A MWE of loading a set of aligned video frames from a kwcoco dataset.

SeeAlso:
    <watch-repo>/examples/mwe_ndsampler_load_region.py

"""
import watch
import kwcoco
import numpy as np


coco_fpath = 'watch-msi'  # demo data code

if 0:
    # To use project data ensure this path is appropriately set.
    # See: geowatch_dvc --help to register where your watch data dvc repo is.
    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = dvc_data_dpath / 'Drop4-BAS/data_vali.kwcoco.json'

dset : kwcoco.CocoDataset = watch.coerce_kwcoco()


# Loop over videos
for video_id in dset.videos():
    print(f'video_id={video_id}')

    # Loop over images in the video
    for image_id in dset.images(video_id=video_id):
        print(f'image_id={image_id}')

        coco_image : kwcoco.CocoImage =  dset.coco_image(image_id)

        # Get a delayed image reference in "video space"
        delayed_im = coco_image.imdelay(space='video')

        # Do any scaling / cropping you want here in the aligned video space
        delayed_im = delayed_im.scale(0.5)
        delayed_im = delayed_im.crop((slice(64, 256), slice(32, 256)))

        # After applying any transforms / crops, finalize the image and execute
        # the optimized operations. (Leverages COG tiles and overviews)
        im : np.ndarray = delayed_im.finalize()
        print(f'im.shape={im.shape}')
