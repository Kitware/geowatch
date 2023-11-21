import ubelt as ub
import scriptconfig as scfg
import os


class CocoReformatChannels(scfg.DataConfig):
    """
    Helper to remove channels from a coco file and reformat predictions from
    float32 to int16.
    """
    __default__ = {
        'src': scfg.Value(None, help='path to coco dataset to read and rewrite'),

        'quantize': scfg.Value(None, type=str, help='channels to quantize'),

        'remove': scfg.Value(None, type=str, help='channels to remove'),

        'workers': scfg.Value(0, help='number of background workers'),

        'nodata': scfg.Value(-9999, help='nodata value to use'),
    }


def main(cmdline=False, **kwargs):
    """
    Example:
        >>> from geowatch.cli.coco_reformat_channels import *  # NOQA
        >>> import kwcoco
        >>> import kwarray
        >>> import shutil
        >>> # Make a dataset we can modify inplace
        >>> orig_dset = kwcoco.CocoDataset.demo('vidshapes1-msi')
        >>> orig_dset.reroot(absolute=False)
        >>> orig_bundle = ub.Path(orig_dset.bundle_dpath)
        >>> new_bundle = ub.Path.appdir('kwcoco/tests/test_reformat_channels')
        >>> new_bundle.delete().ensuredir()
        >>> shutil.copytree(orig_bundle, new_bundle, dirs_exist_ok=True)
        >>> new_fpath = new_bundle / 'data.kwcoco.json'
        >>> orig_dset.dump(new_fpath)
        >>> new_dset = kwcoco.CocoDataset(new_fpath)
        >>> gid = 1
        >>> coco_img = new_dset.coco_image(gid)
        >>> rng = kwarray.ensure_rng(432)
        >>> imdata = rng.rand(128, 128)
        >>> coco_img.add_auxiliary_item('test_prediction.tif', channels='salient', imdata=imdata, imwrite=True)
        >>> new_dset.dump(new_dset.fpath)
        >>> # now reformat this new dataset
        >>> orig_pred = coco_img.imdelay('salient').finalize()
        >>> kwargs = {
        >>>     'src': new_dset.fpath,
        >>>     'quantize': 'salient',
        >>>     #'remove': 'B11',
        >>> }
        >>> cmdline = False
        >>> main(cmdline=False, **kwargs)
        >>> reformatted_dset = kwcoco.CocoDataset(new_fpath)
        >>> assert 'quantization' in reformatted_dset.imgs[1]['auxiliary'][-1]
        >>> new_coco_img = reformatted_dset.coco_image(gid)
        >>> import numpy as np
        >>> new_pred1 = np.nan_to_num(new_coco_img.imdelay('salient').finalize())
        >>> #assert np.allclose(new_pred1, new_pred2)
        >>> #new_pred2 = new_coco_img.imdelay('salient').finalize(dequantize=False)
        >>> #assert new_pred2.dtype.kind == 'i'
    """
    config = CocoReformatChannels.cli(data=kwargs, cmdline=cmdline, strict=True)
    import kwcoco
    print('config = {}'.format(ub.urepr(config, nl=1)))
    dset = kwcoco.CocoDataset.coerce(config['src'])

    to_quantize = kwcoco.ChannelSpec.coerce('' if config['quantize'] is None else config['quantize']).fuse()
    to_remove = kwcoco.ChannelSpec.coerce('' if config['remove'] is None else config['remove']).fuse()

    tasks = []
    for coco_img in dset.images().coco_images:
        for stream in coco_img.channels.streams():
            task = {}
            has_quantize = stream & to_quantize
            has_remove = stream & to_remove
            obj = coco_img.find_asset_obj(stream)
            old_quantization = obj.get('quantization', None)
            if has_remove.numel():
                has_quantize = has_quantize - has_remove
            if has_remove is not None and has_remove.numel():
                task['has_remove'] = has_remove
            if old_quantization is None:
                # Dont redo quantization
                if has_quantize.numel():
                    total_modified = has_quantize.as_oset() | has_remove.as_oset()
                    task['has_quantize'] = has_quantize
                    if len(total_modified) != len(stream):
                        missing = stream - total_modified
                        msg = f'Must quantize or remove entire stream: {stream=!s}, {has_quantize=!s}, {has_remove=!s}, {missing=!s}'
                        task['error'] = msg
            if task:
                task['gid'] = coco_img.img['id']
                task['stream'] = stream
                tasks.append(task)

    if len(tasks) == 0:
        print('No modifications need to be made')
    else:
        print('Found {} tasks'.format(len(tasks)))
        errors = [t['error'] for t in tasks if t.get('error', None)]
        if errors:
            for msg in errors:
                print(msg)
            raise ValueError('Cannot modify')

        gid_to_tasks = ub.group_items(tasks, lambda x: x['gid'])

        bundle_dpath = ub.Path(dset.bundle_dpath)
        jobs = ub.JobPool('process', max_workers=config['workers'])

        for gid, sub_tasks in ub.ProgIter(list(gid_to_tasks.items()), desc='submit reformat jobs'):
            coco_img = dset.coco_image(gid)
            for task in sub_tasks:
                stream = task['stream']
                has_remove = task.get('has_remove', None)
                has_quantize = task.get('has_quantize', None)
                obj = coco_img.find_asset_obj(stream)

                job = jobs.submit(reformat_obj, obj, bundle_dpath, has_remove,
                                  has_quantize)
                job.obj = obj

        finalize_tasks = []
        for job in jobs.as_completed(desc='reformat images'):
            fpath, new_fpath, new_obj = job.result()
            # Update the coco object in memory
            job.obj.update(new_obj)
            finalize_tasks.append({
                'dst': fpath,
                'src': new_fpath,
            })

        tmp_fpath = ub.Path(dset.fpath).augment(suffix='.movtmp')
        dset.dump(tmp_fpath, newlines=True)

        # Point of no return

        # Finalize everything by executing move
        os.rename(tmp_fpath, dset.fpath)
        for task in ub.ProgIter(finalize_tasks, desc='finalize image move'):
            os.rename(task['src'], task['dst'])


def reformat_obj(obj, bundle_dpath, has_remove, has_quantize):
    import xarray as xr
    from kwcoco.channel_spec import FusedChannelSpec
    import kwimage
    import numpy as np
    quantize_dtype = np.int16
    quantize_max = np.iinfo(quantize_dtype).max
    quantize_min = 0
    quantize_nan = -9999

    fname = ub.Path(obj['file_name'])
    fpath = bundle_dpath / fname

    # Write to a temporary sidecar so we dont clobber anything until
    # we know the entire operation worked
    new_fpath = fpath.augment(suffix='.modtmp')

    # Might be able to do this with a gdal command instead
    ma_imdata = kwimage.imread(fpath, nodata='ma')
    imdata = ma_imdata.data
    mask = ma_imdata.mask
    new_obj = obj.copy()

    assert obj['channels'] is not None
    obj_channels = FusedChannelSpec.coerce(obj['channels'])

    if has_remove is not None and has_remove.numel():
        keep_channels = obj_channels - has_remove
        coords = {}
        if obj_channels is not None:
            coords['c'] = obj_channels.as_list()
        xr_imdata = xr.DataArray(imdata, dims=('y', 'x', 'c'), coords=coords)
        xr_mask = xr.DataArray(mask, dims=('y', 'x', 'c'), coords=coords)
        imdata_ = xr_imdata.loc[:, :, keep_channels].data
        mask_ = xr_mask.loc[:, :, keep_channels].data
        new_obj.update({
            'channels': keep_channels.concise().spec,
            'num_bands': keep_channels.numel(),
        })
    else:
        keep_channels = obj_channels
        imdata_ = imdata
        mask_ = mask

    prev_quant = obj.get('quantization', None)
    if prev_quant is None and has_quantize is not None and has_quantize.numel():
        assert prev_quant is None
        assert has_quantize.as_set() == keep_channels.as_set()
        if imdata.dtype.kind == 'f':
            max_val = ma_imdata.max()
            min_val = ma_imdata.min()
            if min_val < 0 or max_val > 1:
                raise Exception('Can only quantize 0-1 float arrays')
            old_max = 1
            old_min = 0
        else:
            raise NotImplementedError

        old_extent = (old_max - old_min)
        new_extent = (quantize_max - quantize_min)
        quant_factor = new_extent / old_extent

        quantization = {
            'orig_min': old_min,
            'orig_max': old_max,
            'quant_min': quantize_min,
            'quant_max': quantize_max,
            'nodata': quantize_nan,
        }
        new_imdata = (imdata_ - old_min) * quant_factor + quantize_min
        new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[mask_] = quantize_nan

        # Denote that this channel is quantized
        new_obj.update({
            'quantization': quantization,
        })
    else:
        new_imdata = imdata_

    kwimage.imwrite(new_fpath, new_imdata, backend='gdal',
                    compress='DEFLATE', blocksize=128,
                    nodata=quantize_nan)

    return fpath, new_fpath, new_obj


def schedule_quantization():
    # Temporary job
    import geowatch
    dvc_dpath = ub.Path(str(geowatch.find_dvc_dpath()) + '-hdd')
    dvc_dpath / 'models/fusion/'

    pred_globpat = dvc_dpath / 'models/fusion/Drop2-Aligned-TA1-2022-02-15/*/pred_*/*/pred.kwcoco.json'
    # pred_globpat = dvc_dpath / 'models/fusion/Drop2-Aligned-TA1-2022-02-15/*/*/*/pred.kwcoco.json'

    import glob
    pred_fpaths = glob.glob(str(pred_globpat))
    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', name='quantize-jobs', size=8)
    for pred_fpath in pred_fpaths:
        queue.submit(ub.codeblock(
            rf'''
            python -m geowatch.cli.coco_reformat_channels \
                --src={pred_fpath} \
                --quantize="not_salient|salient,Site Preparation|Active Construction|Post Construction|No Activity" \
                --remove="not_salient,negative,ignore,negative,positive,background|Unknown" \
                --workers=0
            '''))
    queue.rprint()
    queue.run()
    queue.monitor()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/coco_combine_features.py
    """
    main(cmdline=True)
