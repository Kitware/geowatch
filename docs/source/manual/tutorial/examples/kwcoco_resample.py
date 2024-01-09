"""
Is there an easy way to resample a KWCOCO regions? Like generate a 300x300
image of each project region?

Yes, here is an example where we completely rewrite everything in a fixed
resolution.
"""


def kwcoco_resample_example():
    import watch
    import numpy as np
    import kwimage
    import ubelt as ub
    import kwcoco

    USE_TOYDATA = 0
    if USE_TOYDATA:
        src_coco_dset = watch.coerce_kwcoco()
    else:
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        kwcoco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'
        src_coco_dset = kwcoco.CocoDataset(kwcoco_fpath)

    # We will construct a new kwcoco bundle that contains the resampled data.
    # Note: resampling can be done on the fly using the methods illustrated in
    # this example, but completely rewriting the data is an extreme use case
    # that demonstrates this.
    src_coco_fpath = ub.Path(src_coco_dset.fpath)

    src_bundle_dpath = src_coco_fpath.parent
    dst_bundle_dpath = src_bundle_dpath.augment(tail='_resampled')
    dst_bundle_dpath.ensuredir()
    dst_bundle_fpath = dst_bundle_dpath / src_coco_fpath.name

    dst_coco_dset = kwcoco.CocoDataset()
    dst_coco_dset.fpath = dst_bundle_fpath

    # We want all of our videos to be this new width / height
    dst_video_dsize = np.array([300, 300])

    # Copy over categories
    for cat in src_coco_dset.index.cats.values():
        dst_coco_dset.add_category(**cat)

    # Loop through all videos / images like:
    for src_video_id in src_coco_dset.videos():

        src_video_obj: dict = src_coco_dset.index.videos[src_video_id]

        # Get the current video width / height
        src_video_dsize = np.array([src_video_obj['width'], src_video_obj['height']])

        # We need to construct a transform from the old video space to the new
        # video space. The kwimage.Affine object makes this easy.
        scale_dst_from_src = dst_video_dsize / src_video_dsize
        warp_dst_from_src = kwimage.Affine.scale(scale_dst_from_src)

        # Also get an approximate scalar scale
        approx_scale = scale_dst_from_src.mean()

        # Copy and update the video metadata
        dst_video_obj = src_video_obj.copy()
        dst_video_obj['width'] = int(dst_video_dsize[0])
        dst_video_obj['height'] = int(dst_video_dsize[1])

        # Take care when updating transform metadata. These are not kwcoco
        # standards, but we use them in watch.
        src_warp_vid_from_wld = kwimage.Affine.coerce(src_video_obj['warp_wld_to_vid'])
        dst_warp_vid_from_wld = warp_dst_from_src @ src_warp_vid_from_wld
        dst_video_obj['dst_warp_wld_to_vid'] = dst_warp_vid_from_wld.concise()
        dst_video_obj['target_gsd'] = float(src_video_obj['target_gsd'] * approx_scale)

        # Add the new video to the new coco dataset
        dst_video_id = dst_coco_dset.add_video(**dst_video_obj)

        for src_image_id in src_coco_dset.images(video_id=src_video_id):

            # Construct a CocoImage object, which will allow us to sample
            # images in any specified resolution.
            src_coco_img : kwcoco.CocoImage = src_coco_dset.coco_image(src_image_id)

            # Take care to separate categorical from continuous raster bands
            # and use proper interpolation on both.
            all_channels = src_coco_img.channels.fuse()
            categorical_bands = all_channels & kwcoco.FusedChannelSpec.coerce('quality|cloudmask')
            continuous_bands = all_channels - categorical_bands

            # Load the old delayed image in videospace and do a delayed warp
            # into the dst video / image space (which will be the same)
            src_delayed_cat = src_coco_img.imdelay(space='video', channels=categorical_bands)
            dst_delayed_cat = src_delayed_cat.warp(warp_dst_from_src, dsize=tuple(dst_video_dsize.tolist()))

            src_delayed_con = src_coco_img.imdelay(space='video', channels=continuous_bands)
            dst_delayed_con = src_delayed_con.warp(warp_dst_from_src, dsize=tuple(dst_video_dsize.tolist()))

            # Use some hacked domain knowledge to handle the right nodata value
            # (The delayed image library could be better about this).
            nodata_value = -9999
            dst_cat_im = dst_delayed_cat.finalize(nodata_method='float', antialias=False, interpolation='nearest', nodata_value=nodata_value)
            dst_con_im = dst_delayed_con.finalize(nodata_method='float', antialias=True, interpolation='cubic', nodata_value=nodata_value)

            # Reformat the data as int16 like it originally was.
            dst_con_im[np.isnan(dst_con_im)] = nodata_value
            dst_con_im = dst_con_im.astype(np.int16)
            dst_cat_im[np.isnan(dst_cat_im)] = nodata_value
            dst_cat_im = dst_cat_im.astype(np.int16)

            # Build a lookup table so we can get the raster image for any band.
            band_to_raster = {}
            for band, im in zip(categorical_bands, dst_cat_im.transpose(2, 0, 1)):
                band_to_raster[band] = im
            for band, im in zip(continuous_bands, dst_con_im.transpose(2, 0, 1)):
                band_to_raster[band] = im

            # Note: the API for constructing a new coco image with auxiliary /
            # assets objects could be improved.
            src_img_obj = src_coco_img.img
            dst_img_obj = src_img_obj.copy()
            dst_assets = dst_img_obj.pop('auxiliary')

            # Update main image metadata
            dst_valid_region = kwimage.MultiPolygon.coerce(src_img_obj['valid_region']).warp(warp_dst_from_src)
            dst_img_obj['valid_region'] = dst_valid_region.to_coco(style='new')
            dst_img_obj['width'] = int(dst_video_dsize[0])
            dst_img_obj['height'] = int(dst_video_dsize[1])

            # Can hack in an identity matrix here.
            warp_vid_from_img = kwimage.Affine.eye()
            dst_img_obj['warp_img_to_vid'] = warp_vid_from_img.concise()

            # Not sure if this propery is used anymore?
            warp_img_from_vid = warp_vid_from_img.inv()
            warp_img_from_wld = warp_img_from_vid @ dst_warp_vid_from_wld
            dst_img_obj['wld_to_pxl'] = warp_img_from_wld.concise()

            for asset in dst_assets:
                # We can hack in an identity here
                warp_img_from_asset = kwimage.Affine.eye()

                warp_asset_from_img = warp_img_from_asset.inv()
                warp_asset_from_wld = warp_asset_from_img @ warp_img_from_wld
                warp_wld_from_asset = warp_asset_from_wld.inv()

                # Update transforms in the asset / auxiliary dictionary
                asset['width'] = int(dst_video_dsize[0])
                asset['height'] = int(dst_video_dsize[1])
                asset['warp_aux_to_img'] = warp_img_from_asset.concise()
                asset['warp_to_wld'] = warp_wld_from_asset.concise()
                asset['wld_to_pxl'] = warp_asset_from_wld.concise()

                # Lookup the raster for this new image
                band = asset['channels']

                # Lookup the image data that needs to be written
                new_data = band_to_raster[band]

                # Determine where we will write it
                fname = asset['file_name']
                new_fpath = dst_bundle_dpath / fname
                old_fpath = src_bundle_dpath / fname

                assert new_fpath != old_fpath, 'dont overwrite your old bundle!'

                # We will write our new images as geotiffs and handle geotiff
                # metadata (not necessary, but makes other peoples lives
                # easier)
                write_kwargs = {}
                write_kwargs['blocksize'] = 128
                write_kwargs['compress'] = 'DEFLATE'
                write_kwargs['nodata'] = nodata_value
                from osgeo import osr
                auth = asset['wld_crs_info']['auth']
                assert auth[0] == 'EPSG', 'unhandled auth'
                epsg = auth[1]
                axis_strat = getattr(osr, asset['wld_crs_info']['axis_mapping'])
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(int(epsg))
                srs.SetAxisMappingStrategy(axis_strat)
                write_kwargs['crs'] = srs.ExportToWkt()
                write_kwargs['transform'] = warp_wld_from_asset
                write_kwargs['overviews'] = 2

                # Ensure the directory structure exists and write the new asset
                new_fpath.parent.ensuredir()

                kwimage.imwrite(
                    new_fpath, new_data, backend='gdal', **write_kwargs,
                )

        # Update the image with the new video id and warped assets
        dst_img_obj['auxiliary'] = dst_assets
        dst_img_obj['video_id'] = dst_video_id

        new_image_id = dst_coco_dset.add_image(**dst_img_obj)

        # Bring over all the old annotations (although we could also just
        # reproject them from the original site model files).
        # Annots are in image space, so warp them accordingly

        src_annots = src_coco_dset.annots(gid=src_coco_img.img['id'])
        for src_annot_obj in src_annots.objs:
            dst_annot_obj = src_annot_obj.copy()

            src_poly = kwimage.MultiPolygon.coerce(dst_annot_obj['segmentation'])
            src_bbox = kwimage.Boxes([dst_annot_obj['bbox']], 'xywh')

            # We can ignore the fact that this transform is in video space
            # because we know the warp between video and image space in the new
            # dataset is the identity. A better example would account for this.
            dst_poly = src_poly.warp(warp_dst_from_src)
            dst_bbox = src_bbox.warp(warp_dst_from_src)

            dst_annot_obj['segmentation'] = dst_poly.to_coco(style='new')
            dst_annot_obj['bbox'] = list(dst_bbox.to_coco())[0]
            dst_annot_obj['image_id'] = new_image_id

            dst_coco_dset.add_annotation(**dst_annot_obj)

    # Write the new warped kwcoco file
    dst_coco_dset.dump()
