r"""
Writing cold kwcoco script after predict.py

CommandLine:

    #######################
    ### FULL REGION TEST-V1
    #######################

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m geowatch.tasks.cold.writing_kwcoco \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly-KR_R001.kwcoco.json" \
        --combined_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip" \
        --out_dpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/_pycold_combine_V1" \
        --mod_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold-V1.kwcoco.zip" \
        --method='COLD' \
        --timestamp=False \
        --combine=True \
        --resolution='10GSD' \
        --workermode='serial' \
        --workers=0

    kwcoco stats "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold-V1.kwcoco.zip
    geowatch stats "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold-V1.kwcoco.zip
    kwcoco validate "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold-V1.kwcoco.zip

    geowatch visualize \
        "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold-V1.kwcoco.zip \
        --channels="L8:(red|green|blue,red_COLD_a1|green_COLD_a1|blue_COLD_a1,red_COLD_cv|green_COLD_cv|blue_COLD_cv,red_COLD_rmse|green_COLD_rmse|blue_COLD_rmse)" \
        --exclude_sensors=WV,PD,S2 \
        --smart=True

    #######################
    ### FULL REGION TEST V2
    #######################

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m geowatch.tasks.cold.writing_kwcoco \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly-KR_R001.kwcoco.json" \
        --out_dpath="$DATA_DVC_DPATH//Drop6/_pycold_combine_V2" \
        --mod_coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip" \
        --method='COLD' \
        --timestamp=False \
        --combine=False \
        --resolution='10GSD' \
        --workermode='serial' \
        --workers=0

    kwcoco stats "$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip"
    geowatch stats "$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip"
    kwcoco validate "$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip"

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    geowatch visualize \
        "$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-V2.kwcoco.zip" \
        --channels="L8:(red|green|blue,red_COLD_a1|green_COLD_a1|blue_COLD_a1,red_COLD_cv|green_COLD_cv|blue_COLD_cv,red_COLD_rmse|green_COLD_rmse|blue_COLD_rmse)" \
        --exclude_sensors=WV,PD,S2 \
        --smart=True

"""
import scriptconfig as scfg
import ubelt as ub
import os
import json
import logging
import kwcoco
import kwimage
import pandas as pd

logger = logging.getLogger(__name__)


try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class WriteColdCocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """

    coco_fpath = scfg.Value(None, position=1, help=ub.paragraph(
        '''
        a path to a file to input kwcoco file (to predict on)
        '''))
    combined_coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined input kwcoco file (to merge with)
        '''))
    mod_coco_fpath = scfg.Value(None, help='file path for modified output coco json')
    out_dpath = scfg.Value(None, help='output directory for the output. If unspecified uses the output kwcoco bundle')
    method = scfg.Value('COLD', choices=['COLD', 'HybridCOLD', 'OBCOLD'], help='type of cold algorithms')
    timestamp = scfg.Value(False, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    combine = scfg.Value(True, help='for temporal combined mode, Default is True')
    track_emissions = scfg.Value(True, help='if True use codecarbon for emission tracking')
    resolution = scfg.Value('30GSD', help='if specified then data is processed at this resolution')
    workers = scfg.Value(8, help='total number of workers')
    workermode = scfg.Value('process', help='Can be process, serial, or thread')


@profile
def cold_writing_kwcoco_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m geowatch.tasks.cold.writing_kwcoco --help
        TEST_COLD=1 xdoctest -m geowatch.tasks.cold.writing_kwcoco cold_writing_kwcoco_main

     Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from geowatch.tasks.cold.writing_kwcoco import cold_writing_kwcoco_main
        >>> from geowatch.tasks.cold.writing_kwcoco import *
        >>> kwargs= dict(
        >>>   coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6/imgonly-KR_R001.kwcoco.json'),
        >>>   combined_coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'),
        >>>   mod_coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6/imgonly-KR_R001.kwcoco.testing_comFalse.json'),
        >>>   out_dpath = ub.Path.appdir('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/_pycold_combine1'),
        >>>   method = 'COLD',
        >>>   timestamp = False,
        >>>   combine = False,
        >>>   resolution = '10GSD',
        >>>   workermode = 'process',
        >>> )
        >>> cmdline=0
        >>> cold_writing_kwcoco_main(cmdline, **kwargs)
    """
    #NOTE: This script doesn't consider timestamp = True
    config = WriteColdCocoConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    from geowatch.utils import process_context
    from kwutil import util_json
    resolved_config = config.to_dict()
    resolved_config = util_json.ensure_json_serializable(resolved_config)

    proc_context = process_context.ProcessContext(
        name='geowatch.tasks.cold.writing_kwcoco',
        type='process',
        config=resolved_config,
        track_emissions=config['track_emissions'],
    )

    # Assign variables
    coco_fpath = ub.Path(config['coco_fpath'])
    combine = config['combine']

    if config['combined_coco_fpath'] is not None:
        combined_coco_fpath = ub.Path(config['combined_coco_fpath'])
    else:
        if combine:
            raise ValueError('Must specify combined_coco_fpath if combine is True')
        combined_coco_fpath = None

    if config['out_dpath'] is None:
        config['out_dpath'] = coco_fpath.parent

    mod_coco_fpath = ub.Path(config['mod_coco_fpath'])
    out_dpath = ub.Path(config['out_dpath']).ensuredir()
    method = config['method']
    resolution = config['resolution']
    metadata = read_json_metadata(out_dpath)
    region_id = metadata['region_id']
    cold_feat_path = out_dpath / 'reccg' / region_id / 'cold_feature'

    coef_names = ['cv', 'rmse', 'a0', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'c1']
    band_names = [0, 1, 2, 3, 4, 5]

    BAND_INFO = {0: 'blue',
                 1: 'green',
                 2: 'red',
                 3: 'nir',
                 4: 'swir16',
                 5: 'swir22'}

    proc_context.start()
    proc_context.add_disk_info(out_dpath)

    logger.info('Starting adding new asset to kwcoco json')

    asset_w = metadata['video_w']
    asset_h = metadata['video_h']

    if combine:
        combined_coco_dset = kwcoco.CocoDataset(combined_coco_fpath)
        coco_dset = kwcoco.CocoDataset(coco_fpath)

        for image_id in combined_coco_dset.images():
            combined_coco_image: kwcoco.CocoImage = combined_coco_dset.coco_image(image_id)
            coco_image: kwcoco.CocoImage = coco_dset.coco_image(image_id)
            image_name = coco_image.img['name']

            for band_name in band_names:
                for coef in coef_names:
                    band = BAND_INFO[band_name]
                    cold_feat_fpath = cold_feat_path / f'{image_name}_{band}_{method}_{coef}.tif'
                    if cold_feat_fpath.exists():
                        channels = kwcoco.ChannelSpec.coerce(f'{band}_{method}_{coef}')
                        # COLD output was wrote based on transform information of
                        # coco_dset, so it aligned to a scaled video space.
                        warp_img_from_vid = combined_coco_image.warp_img_from_vid

                        if resolution is None:
                            scale_asset_from_vid = (1., 1.)
                        else:
                            scale_asset_from_vid = combined_coco_image._scalefactor_for_resolution(
                                space='video', resolution=resolution)
                        warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)
                        warp_vid_from_asset = warp_asset_from_vid.inv()
                        warp_img_from_asset = warp_img_from_vid @ warp_vid_from_asset

                        # Use the CocoImage helper which will augment the coco dictionary with
                        # your information.
                        combined_coco_image.add_asset(
                            file_name=cold_feat_fpath,
                            channels=channels,
                            width=asset_w,
                            height=asset_h,
                            warp_aux_to_img=warp_img_from_asset)
                        # logger.info(f'Added to the asset {cold_feat_fpath}')
    else:
        coco_dset = kwcoco.CocoDataset(coco_fpath)

        # Get ordinal day list
        block_folder = out_dpath / 'stacked' / region_id / 'block_x1_y1'

        # if timestamp:
        meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]

        # sort image files by ordinal dates
        img_dates = []
        img_names = []

        # read metadata and get img_name list of first date of each year
        for meta in meta_files:
            meta_config = json.loads((block_folder / meta).read_text())
            ordinal_date = meta_config['ordinal_date']
            img_name = meta_config['image_name']
            img_dates.append(ordinal_date)
            img_names.append(img_name)

        img_dates = sorted(img_dates)
        img_names = sorted(img_names)

        # Get only the first ordinal date of each year
        first_ordinal_dates = []
        first_img_names = []
        last_year = None
        for ordinal_day, img_name in zip(img_dates, img_names):
            year = pd.Timestamp.fromordinal(ordinal_day).year
            if year != last_year:
                first_ordinal_dates.append(ordinal_day)
                first_img_names.append(img_name)
                last_year = year

        img_names = first_img_names

        for image_id in coco_dset.images():
            # Create a CocoImage object for each image.
            coco_image: kwcoco.CocoImage = coco_dset.coco_image(image_id)
            image_name = coco_image.img['name']
            if image_name in img_names:
                for band_name in band_names:
                    for coef in coef_names:
                        band = BAND_INFO[band_name]
                        cold_feat_fpath = cold_feat_path / f'{image_name}_{band}_{method}_{coef}.tif'
                        if cold_feat_fpath.exists():
                            channels = kwcoco.ChannelSpec.coerce(f'{band}_{method}_{coef}')

                            # COLD output was wrote based on transform information of
                            # coco_dset, so it aligned to a scaled video space.
                            warp_img_from_vid = coco_image.warp_img_from_vid

                            if resolution is None:
                                scale_asset_from_vid = (1., 1.)
                            else:
                                scale_asset_from_vid = coco_image._scalefactor_for_resolution(
                                    space='video', resolution=resolution)
                            warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)
                            warp_vid_from_asset = warp_asset_from_vid.inv()
                            warp_img_from_asset = warp_img_from_vid @ warp_vid_from_asset

                            # Use the CocoImage helper which will augment the
                            # coco dictionary with your information.
                            coco_image.add_asset(os.fspath(cold_feat_fpath),
                                                 channels=channels,
                                                 width=asset_w, height=asset_h,
                                                 warp_aux_to_img=warp_img_from_asset)

                            logger.info(f'Added to the asset {cold_feat_fpath}')

    if proc_context is not None:
        context_info = proc_context.stop()
        coco_dset.dataset['info'].append(context_info)

    # Write a modified kwcoco.json file
    logger.info(f'Writing kwcoco file to: {mod_coco_fpath}')
    if combine:
        combined_coco_dset.fpath = mod_coco_fpath
        combined_coco_dset._ensure_json_serializable()
        combined_coco_dset.dump()
    else:
        coco_dset.fpath = mod_coco_fpath
        coco_dset._ensure_json_serializable()
        coco_dset.dump()
    logger.info(f'Finished writing kwcoco file to: {mod_coco_fpath}')


def read_json_metadata(folder_path):
    stacked_path = folder_path / 'stacked'
    for root, dirs, files in os.walk(stacked_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                with open(json_path, "r") as f:
                    metadata = json.load(f)
                    return metadata


if __name__ == '__main__':
    cold_writing_kwcoco_main()
