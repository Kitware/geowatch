r"""
Transfering cold features

CommandLine:

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m watch.tasks.cold.transfer_features \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-HTR.kwcoco.zip" \
        --combine_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip" \
        --new_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip" 
        
"""

import ubelt as ub
import kwcoco
import watch
import kwimage
from os.path import join
from watch.utils import util_time
import scriptconfig as scfg

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class TransferCocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """

    coco_fpath = scfg.Value(None, position=1, help=ub.paragraph(
        '''
        a path to a file to input kwcoco file (to predict on)
        '''))
    combine_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined input kwcoco file (to merge with)
        '''))
    new_coco_fpath = scfg.Value(None, help='file path for modified output coco json')
    channels_to_transfer = scfg.Value(None, help='COLD channels for transfer')
    # sensors = scfg.Value(None, help='type of sensors')
    track_emissions = scfg.Value(True, help='if True use codecarbon for emission tracking')
    workers = scfg.Value(8, help='total number of workers')
    workermode = scfg.Value('process', help='Can be process, serial, or thread')

@profile
def transfer_features_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m watch.tasks.cold.transfer_features --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.transfer_features transfer_features_main

     Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from watch.tasks.cold.writing_kwcoco import transfer_features_main
        >>> from watch.tasks.cold.writing_kwcoco import *
        >>> kwargs= dict(
        >>>   coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip'),
        >>>   combine_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'),
        >>>   new_coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold_test.kwcoco.zip'),
        >>>   workermode = 'process',
        >>> )
        >>> cmdline=0
        >>> cold_writing_kwcoco_main(cmdline, **kwargs)
    """
    #NOTE: This script doesn't consider timestamp = True
    
    config = TransferCocoConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    from watch.utils import process_context
    from watch.utils import util_json
    resolved_config = config.to_dict()
    resolved_config = util_json.ensure_json_serializable(resolved_config)

    proc_context = process_context.ProcessContext(
        name='watch.tasks.cold.transfer_features',
        type='process',
        config=resolved_config,
        track_emissions=config['track_emissions'],
    )

    # Assign variables
    default_channels = ['blue_COLD_cv','green_COLD_cv','red_COLD_cv','nir_COLD_cv', 'swir16_COLD_cv', 'swir22_COLD_cv', 'blue_COLD_a0', 'green_COLD_a0', 'red_COLD_a0', 'nir_COLD_a0', 'swir16_COLD_a0', 'swir22_COLD_a0', 'blue_COLD_a1', 'green_COLD_a1', 'red_COLD_a1', 'nir_COLD_a1', 'swir16_COLD_a1','swir22_COLD_a1','blue_COLD_b1','green_COLD_b1','red_COLD_b1','nir_COLD_b1','swir16_COLD_b1','swir22_COLD_b1','blue_COLD_c1','green_COLD_c1','red_COLD_c1','nir_COLD_c1','swir16_COLD_c1','swir22_COLD_c1','blue_COLD_rmse','green_COLD_rmse','red_COLD_rmse','nir_COLD_rmse','swir16_COLD_rmse','swir22_COLD_rmse']

    coco_fpath = ub.Path(config['coco_fpath'])
    combine_fpath = ub.Path(config['combine_fpath'])
    new_coco_fpath = ub.Path(config['new_coco_fpath'])

    if config['channels_to_transfer'] is None:
        channels_to_transfer = default_channels
    else:        
        channels_to_transfer = config['channels_to_transfer']
        channels_to_transfer = list(channels_to_transfer)

    src = kwcoco.CocoDataset(coco_fpath)
    dst = kwcoco.CocoDataset(combine_fpath)

    src_video_names = src.videos().lookup('name')

    # We will build a list containing all of the assignments to make
    assignments = []

    for vidname in src_video_names:
        # For each corresponding video
        src_video = src.index.name_to_video[vidname]
        dst_video = dst.index.name_to_video[vidname]
        # Look at each sequence of images
        all_src_images = src.images(video_id=src_video['id'])
        dst_images = dst.images(video_id=dst_video['id'])                
        
        # Filter out the source images missing the channels we want to transfer
        keep_flags = [
            coco_img.channels.intersection(channels_to_transfer)
            for coco_img in all_src_images.coco_images
        ]
        src_images = all_src_images.compress(keep_flags)
        
        # Group images by sensor, so we only transfer between similar sensors
        src_sensors = src_images.lookup('sensor_coarse')
        dst_sensors = dst_images.lookup('sensor_coarse')
        sensor_to_dst_gids = ub.group_items(dst_images, dst_sensors)
        sensor_to_src_gids = ub.group_items(src_images, src_sensors)

        for sensor in sensor_to_src_gids.keys():
            src_image_ids = sensor_to_src_gids[sensor]
            dst_image_ids = sensor_to_dst_gids[sensor]

            src_subimgs = src.images(src_image_ids)
            dst_subimgs = dst.images(dst_image_ids)

            # Find the timestamps for each image in both sequences
            src_timestamps = [util_time.coerce_datetime(d) for d in src_subimgs.lookup('date_captured')]
            dst_timestamps = [util_time.coerce_datetime(d) for d in dst_subimgs.lookup('date_captured')]
            assert sorted(src_timestamps) == src_timestamps, 'data should be in order'
            assert sorted(dst_timestamps) == dst_timestamps, 'data should be in order'

            # We will use logic in reproject annotations to assign images with COLD
            # features to the nearest image in the future. In this case the
            # keyframes are the images with COLD features and the target sequence
            # are the images we are transfering onto.
            target_times = dst_timestamps
            
            key_infos = [{'time': d, 'applies': 'future'} for d in src_timestamps]
            from watch.cli.reproject_annotations import keyframe_interpolate
            assigned_indexes = keyframe_interpolate(target_times, key_infos)
            # The above function returns a list for each key-frame with the
            # assigned indexes but we are only going to assign it to one of them,
            # so choose the earliest one in each group. The minimum index will be
            # the earliest because dates are sorted.
            for src_idx, dst_idxs in enumerate(assigned_indexes):
                if dst_idxs:
                    assignments.append({
                        'src_image_id': src_image_ids[src_idx],
                        'dst_image_id': dst_image_ids[min(dst_idxs)],
                        'video_name': vidname,
                    })


    # Now we have all assignments
    vidname_to_assignments = ub.group_items(assignments, lambda x: x['video_name'])
    for vidname, vid_assignments in vidname_to_assignments.items():
        src_video = src.index.name_to_video[vidname]
        dst_video = dst.index.name_to_video[vidname]
        # In most cases the source and destination videos will have the exact same
        # width / height, but in some cases they may differ if one is downsampled.
        # Just to be safe, assume the videos align and compute a scale factor
        # transform to account for this case (its just the identity if width/height
        # are the same).
        fx = src_video['width'] / src_video['width']
        fy = dst_video['height'] / dst_video['height']
        warp_dstvid_from_srcvid = kwimage.Affine.scale((fx, fy))

        for assignment in vid_assignments:
            src_image_id = assignment['src_image_id']
            dst_image_id = assignment['dst_image_id']
            src_coco_img = src.coco_image(src_image_id)
            dst_coco_img = dst.coco_image(dst_image_id)

            # Compute the alignment between assigned images.
            warp_srcvid_from_srcimg = src_coco_img.warp_vid_from_img
            warp_dstimg_from_dstvid = dst_coco_img.warp_img_from_vid
            warp_dstimg_from_srcimg = (
                warp_dstimg_from_dstvid @
                warp_dstvid_from_srcvid @
                warp_srcvid_from_srcimg
            )

            # For each asset in the src, check if we want to transfer it.
            assets_to_transfer = []
            for asset in src_coco_img.iter_asset_objs():
                asset_channels = kwcoco.FusedChannelSpec.coerce(asset['channels'])
                if asset_channels.intersection(channels_to_transfer):                
                    assets_to_transfer.append(asset)
            for asset in assets_to_transfer:
                new_asset = asset.copy()
                # Ensure the filename points to the correct place (just use the
                # absolute path for simplicity for now)
                new_asset['file_name'] = join(src.bundle_dpath, asset['file_name'])
                # Handle the case where the transform is not identity
                warp_srcimg_from_aux = kwimage.Affine.coerce(asset['warp_aux_to_img'])
                warp_dstimg_from_aux = warp_dstimg_from_srcimg @ warp_srcimg_from_aux
                new_asset['warp_aux_to_img'] = warp_dstimg_from_aux.concise()
                dst_coco_img.add_asset(**new_asset)

    # new_coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold_test.kwcoco.zip')
    dst.fpath = new_coco_fpath
    dst._ensure_json_serializable()
    dst.dump()
    
if __name__ == '__main__':
    transfer_features_main()