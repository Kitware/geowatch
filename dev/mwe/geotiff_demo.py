"""
kwimage transform docs:

    https://kwimage.readthedocs.io/en/release/kwimage.html#kwimage.Affine

"""
import numpy as np
import kwimage
import watch
from osgeo import gdal
dates = True
geodata = True
heatmap = True
kwargs = {}
coco_dset = watch.coerce_kwcoco(
    data='watch-msi', dates=dates, geodata=geodata, heatmap=heatmap)

coco_img = coco_dset.images().coco_images[0]

image_fpath = coco_img.primary_image_filepath()

# Do something to make a prediction in "VIDEO SPACE"
vid_w = coco_dset.index.videos[coco_img.img['video_id']]['width']
vid_h = coco_dset.index.videos[coco_img.img['video_id']]['height']
new_vidspace_im = ((np.random.rand(*(vid_h, vid_w))) * 100).astype(np.uint16)


# Now write out the videospace image with correct updated transforms.

ref_image = gdal.Open(image_fpath, gdal.GA_ReadOnly)
trans = ref_image.GetGeoTransform()
proj = ref_image.GetProjection()
cols = ref_image.RasterXSize
rows = ref_image.RasterYSize


# Get original transform from projection to image space
c, a, b, f, d, e = trans
original = kwimage.Affine(np.array([
    [a, b, c],
    [d, e, f],
    [0, 0, 1],
]))


# Get the modifier transform to move from image space to video space
warp_vid_from_img = kwimage.Affine.coerce(coco_img.img['warp_img_to_vid'])


# Combine transforms to get a new transform that goes
# from the projection to video space
new_geotrans = warp_vid_from_img @ original

# Put coefficients in the right order for gdal
a, b, c, d, e, f, g, h, i = np.array(new_geotrans).ravel().tolist()
new_gdal_transform = (c, a, b, f, d, e)

# Write out the new geotiff
mode_string = 'first_disturbance_map'
outname = 'new_videospace_image.tif'
outfile = outname
outdriver1 = gdal.GetDriverByName("GTiff")
outdata = outdriver1.Create(outfile, vid_w, vid_h, 1, gdal.GDT_Int16)
outdata.GetRasterBand(1).WriteArray(new_vidspace_im)
outdata.FlushCache()
outdata.SetGeoTransform(new_gdal_transform)
outdata.FlushCache()
outdata.SetProjection(proj)
outdata.FlushCache()
