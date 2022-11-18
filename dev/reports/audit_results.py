import ubelt as ub
import numpy as np
import json
import kwcoco
from watch.utils import kwcoco_extensions
from watch.mlops import smart_pipeline


def check_crop():
    """

    smartwatch stats kwcoco_for_bas.json

    source $HOME/code/watch/secrets/secrets
    DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/nov-one/debug13_KIT_TA2_20221121_KR_R001/bas-fusion

    # Execute alignment / crop script
    python -m watch.cli.coco_align_geotiffs \
        --src $DPATH/kwcoco_for_bas.json \
        --dst $DPATH/test-recrop/kwcoco_cropped_bas2.json \
        --regions $DPATH/region_models/KR_R001.geojson \
        --rpc_align_method orthorectify \
        --workers=10 \
        --aux_workers=10 \
        --context_factor=1 \
        --visualize=False \
        --geo_preprop=auto \
        --keep img

    """


def audit():
    """
    Local results:
        /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/eval/trk/package_epoch0_step41.pt.pt/Drop4-BAS_KR_R001.kwcoco/trk_pxl_16f221bd/trk_poly_9f08fb8c/merged/summary2.json

       dev_fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/pred/trk/package_epoch0_step41.pt.pt/Drop4-BAS_KR_R001.kwcoco/trk_pxl_fd9e1a95/pred.kwcoco.json')

       docker login http://registry.smartgitlab.com
       docker pull registry.smartgitlab.com/kitware/watch/ta2:Nov21-debug13
       docker run -it registry.smartgitlab.com/kitware/watch/ta2:Nov21-debug13 bash
    """

    dev_fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/pred/trk/package_epoch0_step41.pt.pt/Drop4-BAS_KR_R001.kwcoco/trk_pxl_16f221bd/trk_poly_9f08fb8c/tracks.kwcoco.json')
    dev_ss_manifest_fpath = (dev_fpath.parent / 'site_summary_tracks_manifest.json')
    dev_manifest_data = json.loads(dev_ss_manifest_fpath.read_text())

    audit_root = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/nov-one/debug13_KIT_TA2_20221121_KR_R001')
    bas_bundle = audit_root / 'bas-fusion'
    bas_fnames = [
        'kwcoco_for_sc.json',
        'cropped_kwcoco_for_bas.json',
        'bas_fusion_kwcoco_tracked.json',
        'bas_fusion_kwcoco.json',
        'kwcoco_for_bas.json',
    ]
    bas_fpaths = {n.split('.')[0]: bas_bundle / n for n in bas_fnames}
    pro_tracked_dset = kwcoco.CocoDataset(bas_fpaths['bas_fusion_kwcoco_tracked'])
    pro_input_dset = kwcoco.CocoDataset(bas_fpaths['cropped_kwcoco_for_bas'])

    pro_tracked_dset.reroot(old_prefix='/tmp/ingress/', new_prefix='', absolute=False, verbose=3)
    pro_tracked_dset.validate()

    ### COMPARE TO
    dev_dset = kwcoco.CocoDataset(dev_fpath)

    ### Start AUDIT
    pro_input_audit_info = audit_dataset(pro_input_dset)
    pro_tracked_audit_info = audit_dataset(pro_tracked_dset)
    dev_audit_info = audit_dataset(dev_dset)

    pro_pxl_params = ub.udict(pro_tracked_audit_info['pred_pxl_parms']['pxl'])
    dev_pxl_params = ub.udict(dev_audit_info['pred_pxl_parms']['pxl'])
    flag, check_info = ub.indexable_allclose(pro_pxl_params, dev_pxl_params, return_info=True)
    n_matching_params = len(check_info['passlist'])
    print('matching_params: ' + ub.repr2(check_info['passlist']))
    print(f'n_matching_params={n_matching_params}')
    for k, v1, v2 in check_info['faillist']:
        print(f'Diff {k}: production={v1} -vs- developer={v2}')

    common_keys = dev_pxl_params.keys() & pro_pxl_params.keys()
    for key in common_keys:
        pass

    bas_model_fpath = dev_pxl_params['pxl.package_fpath']

    dev_input_dset_fpath = dev_pxl_params['pxl.test_dataset']
    dev_input_dset = kwcoco.CocoDataset(dev_input_dset_fpath)

    #### Pixel Prediciton AUDIT
    smart_pipeline.parse_tracker_params(dev_manifest_data['info'])

    # Rerun pixel prediction on the production dataset
    trk_pxl_params_parsed = ub.udict({
        k.split('.')[1]: v for k, v in pro_pxl_params.items() if 'properties' not in k
    }) - {'test_dataset', 'package_fpath'}
    trk_pxl_params_parsed['chip_dims'] = ','.join(list(map(str, trk_pxl_params_parsed['chip_dims'])))

    from watch.mlops import schedule_evaluation
    audit_fpath = (dev_fpath.parent / 'audit').ensuredir()

    redev_audit_fpath_list = [
        audit_fpath / 'redo_dev_v1' / 'redo_dev_v1.kwcoco.json',
        audit_fpath / 'redo_dev_v2' / 'redo_dev_v2.kwcoco.json',
        audit_fpath / 'redo_dev_v3' / 'redo_dev_v3.kwcoco.json',
    ]
    repro_audit_fpath_list = [
        audit_fpath / 'repro_v1' / 'pred.kwcoco.json',
        audit_fpath / 'repro_v2' / 'pred.kwcoco.json',
        audit_fpath / 'repro_v3' / 'pred.kwcoco.json',
    ]
    perf_params = {
        'trk.pxl.batch_size': 1,
        'trk.pxl.workers': 2,
        'trk.pxl.devices': '0,',
        'trk.pxl.accelerator': 'gpu',
    }
    trk_pxl_params = trk_pxl_params_parsed
    paths = ub.udict({
        'trk_test_dataset_fpath': pro_input_dset.fpath,
        'pkg_trk_pxl_fpath': bas_model_fpath,
        # 'pred_trk_pxl_fpath': dev_audit_fpath,
    })

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', gres=[0, 1], size=2)

    # ReRun predictions on development inputs
    for pred_fpath in redev_audit_fpath_list:
        paths = paths.copy() | {
            'pred_trk_pxl_fpath': pred_fpath,
            'trk_test_dataset_fpath': dev_input_dset_fpath,
            # dev_dset.fpath
        }
        step = schedule_evaluation.Pipeline.pred_trk_pxl(perf_params, trk_pxl_params, **paths)
        step.otf_cache = 1
        queue.submit(step.command)

    # ReRun predictions on production inputs
    for pred_fpath in repro_audit_fpath_list:
        paths = paths.copy() | {
            'pred_trk_pxl_fpath': pred_fpath,
            'trk_test_dataset_fpath': pro_input_dset.fpath}
        step = schedule_evaluation.Pipeline.pred_trk_pxl(perf_params, trk_pxl_params, **paths)
        step.otf_cache = 1
        queue.submit(step.command)

    queue.rprint()
    queue.run()

    from watch.cli import coco_intensity_histograms
    # Check input histograms

    dev_input_results = coco_intensity_histograms.main(**({
        'src': dev_input_dset_fpath,
        'dst': bas_bundle / 'dev_input_hist.png',
        'workers': 4,
    }))

    pro_input_results = coco_intensity_histograms.main(**({
        'src': pro_input_dset.fpath,
        'dst': bas_bundle / 'pro_input_hist.png',
        'workers': 4,
    }))

    import rich
    rich.print('\n\n[yellow] Development Input Stats:')
    rich.print(dev_input_results['sensor_chan_stats'])
    rich.print('\n\n[green] Production Input Stats:')
    rich.print(pro_input_results['sensor_chan_stats'])

    spectra_defaults = ub.udict({
        'include_channels': 'salient',
        'workers': 4,
    })
    dev_results = coco_intensity_histograms.main(**(spectra_defaults | {
        'src': dev_dset,
        'dst': bas_bundle / 'dev_hist.png'
    }))
    pro_results = coco_intensity_histograms.main(**(spectra_defaults | {
        'src': pro_tracked_dset,
        'dst': bas_bundle / 'pro_hist.png'
    }))

    redev_results = []
    for idx, pred_fpath in enumerate(redev_audit_fpath_list, start=1):
        redev_result = coco_intensity_histograms.main(**(spectra_defaults | {
            'src': pred_fpath,
            'dst': bas_bundle / f'redev_v{idx}_hist.png'
        }))
        redev_results.append(redev_result)

    repro_results = []
    for idx, pred_fpath in enumerate(repro_audit_fpath_list, start=1):
        repro_result = coco_intensity_histograms.main(**(spectra_defaults | {
            'src': pred_fpath,
            'dst': bas_bundle / f'repro_v{idx}_hist.png'
        }))
        repro_results.append(repro_result)

    import rich
    rich.print('\n\n[yellow] Development Stats:')
    rich.print(dev_results['sensor_chan_stats'])

    for idx, results in enumerate(redev_results):
        rich.print(f'\n\n[yellow] Reproduced Development Stats (V{idx}):')
        rich.print(results['sensor_chan_stats'])

    rich.print('\n\n[green] Production Stats:')
    rich.print(pro_results['sensor_chan_stats'])

    for idx, results in enumerate(repro_results):
        rich.print(f'\n\n[green] Reproduced Production Stats (V{idx}):')
        rich.print(results['sensor_chan_stats'])


def check_dataset_differences(dev_input_dset, pro_input_dset, trk_pxl_params, audit_fpath, bas_model_fpath):
    names1 = set(dev_input_dset.index.name_to_img.keys())
    names2 = set(pro_input_dset.index.name_to_img.keys())
    missing1 = names1 - names2
    missing2 = names2 - names1
    import rich
    rich.print('missing1 = {}'.format(ub.repr2(missing1, nl=1)))
    rich.print('missing2 = {}'.format(ub.repr2(missing2, nl=1)))

    ub.Path(pro_input_dset.fpath).parent.ls()
    ub.Path(dev_input_dset.fpath).parent.ls()

    (ub.Path(pro_input_dset.fpath).parent / '_cache').ls()
    (ub.Path(dev_input_dset.fpath).parent / '_cache').ls()

    if 0:
        common_names = names1 & names2
        for name in ub.ProgIter(common_names):
            pro_gid = pro_input_dset.index.name_to_img[name]['id']
            dev_gid = dev_input_dset.index.name_to_img[name]['id']
            coco_img1 = dev_input_dset.coco_image(dev_gid)
            coco_img2 = pro_input_dset.coco_image(pro_gid)

            img_meta1 = ub.udict(coco_img1.img) - {'auxiliary'}
            img_meta2 = ub.udict(coco_img2.img) - {'auxiliary'}
            flag, cmp_info = ub.indexable_allclose(img_meta1, img_meta2, return_info=True)
            img_meta1['valid_region']
            img_meta2['valid_region']
            img_meta1['valid_region_utm']
            img_meta2['valid_region_utm']

            import kwimage
            p1 = kwimage.MultiPolygon.coerce(img_meta1['valid_region'])
            p2 = kwimage.MultiPolygon.coerce(img_meta2['valid_region'])
            iou = p1.intersection(p2).area / p1.union(p2).area
            assert iou == 1

            assets1 = list(coco_img1.iter_asset_objs())
            assets2 = list(coco_img2.iter_asset_objs())

            shared_channels = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']

            ub.dict_hist([a['channels'] for a in assets1])
            ub.dict_hist([a['channels'] for a in assets2])
            channel_to_asset1 = ub.udict({a['channels']: a for a in assets1})
            channel_to_asset2 = ub.udict({a['channels']: a for a in assets2})

            for chan in shared_channels:
                # print(f'Compare: chan={chan}')
                asset1 = channel_to_asset1[chan]
                asset2 = channel_to_asset2[chan]
                flag, cmp_info = ub.indexable_allclose(asset1, asset2, return_info=True)

                fpath1 = ub.Path(dev_input_dset.bundle_dpath) / asset1['file_name']
                fpath2 = ub.Path(pro_input_dset.bundle_dpath) / asset2['file_name']
                hash1 = ub.hash_file(fpath1, hasher='blake3')
                hash2 = ub.hash_file(fpath2, hasher='blake3')
                assert hash1 == hash2

                diff_attrs = cmp_info['faillist']
                diff_attrs = [d for d in diff_attrs if d[0] != ['file_name']]

                if 'passlist' in cmp_info:
                    same_attrs = set([p[0] for p in cmp_info['passlist']])
                    print('Similarities: ' + ub.repr2(same_attrs))

                print('Differences:')
                print(ub.repr2(diff_attrs, nl=2))

            d1 = coco_img1.delay(channels='red|green|blue|nir|swir16|swir22|cloudmask')
            d2 = coco_img2.delay(channels='red|green|blue|nir|swir16|swir22|quality')
            a = d1.finalize()
            b = d2.finalize()
            assert (a - b).sum() == 0

    dev_input_dset_ = dev_input_dset
    pro_input_dset_ = pro_input_dset
    check_datamodule_consistency(dev_input_dset, pro_input_dset, trk_pxl_params)

    images1 = dev_input_dset.videos().images[0]
    images2 = pro_input_dset.videos().images[0]

    dates1 = images1.lookup('date_captured')
    dates2 = images2.lookup('date_captured')

    fx1 = images1.lookup('frame_index')
    fx2 = pro_input_dset.videos().images[0].lookup('frame_index')
    assert fx1 == fx2
    assert sorted(fx1) == fx1
    assert sorted(fx2) == fx2
    # assert sorted(dates1) == dates1
    # assert sorted(dates2) == dates2
    import pandas as pd
    import numpy as np
    cmp_times = pd.DataFrame({'d1': dates1, 'd2': dates2, 'fx1': fx1, 'fx2': fx2})
    print(cmp_times.to_string())
    bad_idxs = np.where((np.array(dates1) != np.array(dates2)))[0]
    print(bad_idxs)
    print(cmp_times.iloc[bad_idxs].to_string())

    dev_input_dset.coco_image(5330).img['date_captured']
    pro_input_dset.coco_image(70).img['date_captured']

    dev_input_stats = dev_input_dset.basic_stats()
    pro_input_stats = pro_input_dset.basic_stats()
    print(f'pro_input_dset.fpath={pro_input_dset.fpath}')
    print(f'dev_input_dset.fpath={dev_input_dset.fpath}')
    print('dev_input_stats = {}'.format(ub.repr2(dev_input_stats, nl=1)))
    print('pro_input_stats = {}'.format(ub.repr2(pro_input_stats, nl=1)))

    bad_gids1 = images1.take(bad_idxs)
    bad_gids2 = images2.take(bad_idxs)

    clean_pro_input_dset = pro_input_dset.copy()
    clean_dev_input_dset = dev_input_dset.copy()

    clean_pro_input_dset.remove_images(bad_gids2)
    clean_dev_input_dset.remove_images(bad_gids1)

    clean_pro_input_dset.reroot(absolute=True)
    clean_dev_input_dset.reroot(absolute=True)

    clean_pro_input_dset.fpath = ub.Path(pro_input_dset.fpath).parent / 'clean_pro_dset.kwcoco.json'
    clean_dev_input_dset.fpath = ub.Path(dev_input_dset.fpath).parent / 'clean_dev_dset.kwcoco.json'

    clean_pro_input_dset.dump(clean_pro_input_dset.fpath)
    clean_dev_input_dset.dump(clean_dev_input_dset.fpath)

    dev_input_dset_ = clean_dev_input_dset
    pro_input_dset_ = clean_pro_input_dset
    check_datamodule_consistency(dev_input_dset_, pro_input_dset_, trk_pxl_params)

    # Try the clean datasets

    import cmd_queue
    from watch.mlops import schedule_evaluation
    queue = cmd_queue.Queue.create(backend='tmux', gres=[0, 1], size=2)

    clean_redev_audit_fpath_list = [
        audit_fpath / 'redo_clean_dev_v1' / 'redo_clean_dev_v1.kwcoco.json',
        # audit_fpath / 'redo_clean_dev_v2' / 'redo_clean_dev_v2.kwcoco.json',
        # audit_fpath / 'redo_clean_dev_v3' / 'redo_clean_dev_v3.kwcoco.json',
    ]
    clean_repro_audit_fpath_list = [
        audit_fpath / 'redo_clean_pro_v1' / 'redo_clean_pro_v1.kwcoco.json',
        # audit_fpath / 'redo_clean_pro_v2' / 'redo_clean_pro_v2.kwcoco.json',
        # audit_fpath / 'redo_clean_pro_v3' / 'redo_clean_pro_v3.kwcoco.json',
    ]
    perf_params = {
        'trk.pxl.batch_size': 1,
        'trk.pxl.workers': 2,
        'trk.pxl.devices': '0,',
        'trk.pxl.accelerator': 'gpu',
    }
    paths = ub.udict({
        # 'trk_test_dataset_fpath': pro_input_dset.fpath,
        'pkg_trk_pxl_fpath': bas_model_fpath,
        # 'pred_trk_pxl_fpath': dev_audit_fpath,
    })

    # ReRun predictions on development inputs
    for pred_fpath in clean_redev_audit_fpath_list:
        paths = paths.copy() | {
            'pred_trk_pxl_fpath': pred_fpath,
            'trk_test_dataset_fpath': clean_dev_input_dset.fpath,
            # dev_dset.fpath
        }
        step = schedule_evaluation.Pipeline.pred_trk_pxl(perf_params, trk_pxl_params, **paths)
        step.otf_cache = 1
        queue.submit(step.command)

    # ReRun predictions on production inputs
    for pred_fpath in clean_repro_audit_fpath_list:
        paths = paths.copy() | {
            'pred_trk_pxl_fpath': pred_fpath,
            'trk_test_dataset_fpath': clean_pro_input_dset.fpath}
        step = schedule_evaluation.Pipeline.pred_trk_pxl(perf_params, trk_pxl_params, **paths)
        step.otf_cache = 1
        queue.submit(step.command)

    queue.rprint()
    queue.write()
    queue.run()

    # prev development: -2349, 185
    # prev production: -2291, 406

    # new development: -2439, 368.6
    # new production: -2441, 373.6

    from watch.cli import coco_intensity_histograms
    spectra_defaults = ub.udict({
        'include_channels': 'salient',
        'workers': 4,
    })
    redev_results = []
    for idx, pred_fpath in enumerate(clean_redev_audit_fpath_list, start=1):
        redev_result = coco_intensity_histograms.main(**(spectra_defaults | {
            'src': pred_fpath,
            # 'dst': bas_bundle / f'redev_v{idx}_hist.png'
        }))
        redev_results.append(redev_result)

    repro_results = []
    for idx, pred_fpath in enumerate(clean_repro_audit_fpath_list, start=1):
        repro_result = coco_intensity_histograms.main(**(spectra_defaults | {
            'src': pred_fpath,
            # 'dst': bas_bundle / f'repro_v{idx}_hist.png'
        }))
        repro_results.append(repro_result)

    import rich
    for idx, results in enumerate(redev_results):
        rich.print(f'\n\n[yellow] Reproduced Development Stats (V{idx}):')
        rich.print(results['sensor_chan_stats'])

    for idx, results in enumerate(repro_results):
        rich.print(f'\n\n[green] Reproduced Production Stats (V{idx}):')
        rich.print(results['sensor_chan_stats'])


def check_datamodule_consistency(dev_input_dset_, pro_input_dset_, trk_pxl_params):
    from watch.tasks.fusion.datamodules import kwcoco_datamodule
    dataset_params = ub.udict(trk_pxl_params) - {'tta_fliprot', 'tta_time'}

    from watch.utils import util_time
    images1 = dev_input_dset_.videos().images[0]
    images2 = pro_input_dset_.videos().images[0]
    dates1 = images1.lookup('date_captured')
    dates2 = images2.lookup('date_captured')

    dt1 = [util_time.coerce_datetime(d) for d in dates1]
    dt2 = [util_time.coerce_datetime(d) for d in dates2]

    np.diff(ub.argsort(dt1)) != 1

    sorted(dt1) == dt1
    sorted(dt2) == dt2

    pro_dmod = kwcoco_datamodule.KWCocoVideoDataModule(**dict(
        test_dataset=pro_input_dset_, **dataset_params,
        use_grid_cache=False,
    ))
    dev_dmod = kwcoco_datamodule.KWCocoVideoDataModule(**dict(
        test_dataset=dev_input_dset_, **dataset_params,
        use_grid_cache=False,
    ))
    pro_dmod.setup('test')
    dev_dmod.setup('test')

    print(len(pro_dmod.test_dataset.new_sample_grid['targets']))
    print(len(dev_dmod.test_dataset.new_sample_grid['targets']))

    pro_sampler = ub.peek(pro_dmod.test_dataset.new_sample_grid['vidid_to_time_sampler'].values())
    dev_sampler = ub.peek(dev_dmod.test_dataset.new_sample_grid['vidid_to_time_sampler'].values())

    pro_sampler.determenistic = True
    dev_sampler.determenistic = True

    for i in range(pro_sampler.num_frames):
        a = pro_sampler.sample(i)
        b = dev_sampler.sample(i)
        assert a == b

    affinity_diff = pro_sampler.affinity - dev_sampler.affinity
    diff_pairs = (affinity_diff > 0).sum()
    print(f'diff_pairs={diff_pairs}')
    import kwarray
    affinity_diff_stats = kwarray.stats_dict(affinity_diff)
    print('affinity_diff_stats = {}'.format(ub.repr2(affinity_diff_stats, nl=1)))
    total_affinity_diff = affinity_diff.sum()
    print(f'total_affinity_diff={total_affinity_diff}')

    is_diff = (dev_sampler.unixtimes - pro_sampler.unixtimes) > 0
    dev_sampler.video_gids[is_diff]
    pro_sampler.video_gids[is_diff]

    from watch.tasks.fusion.datamodules import temporal_sampling
    a = temporal_sampling.soft_frame_affinity(dev_sampler.unixtimes, dev_sampler.sensors, dev_sampler.time_span, version=2)
    b = temporal_sampling.soft_frame_affinity(pro_sampler.unixtimes, pro_sampler.sensors, pro_sampler.time_span, version=2)
    print((a['final'] - b['final']).sum())
    print((dev_sampler.affinity - a['final']).sum())
    print((pro_sampler.affinity - a['final']).sum())

    sensor_diff = np.array(dev_sampler.sensors) != np.array(pro_sampler.sensors)
    print(sensor_diff.sum())

    pro_dmod.test_dataset.new_sample_grid['targets'][0]
    dev_dmod.test_dataset.new_sample_grid['targets'][0]
    pro_dmod.test_dataset.new_sample_grid
    print(len(pro_dmod.test_dataset))
    print(len(dev_dmod.test_dataset))


def audit_dataset(coco_dset):
    chan_stats = kwcoco_extensions.coco_channel_stats(coco_dset)
    sensorchan_hist = chan_stats['sensorchan_hist']
    print('sensorchan_hist = {}'.format(ub.repr2(sensorchan_hist, nl=2)))

    audit_info = {
        'sensorchan_hist': sensorchan_hist,
    }
    info = coco_dset.dataset['info']
    for item in info:
        if isinstance(item, dict) and item.get('type', '') == 'process':
            print(item['properties']['name'])

    try:
        info = coco_dset.dataset['info']
        pred_pxl_parms = smart_pipeline.parse_pred_pxl_params(info)
        audit_info.update({
            'pred_pxl_parms': pred_pxl_parms,
        })
    except Exception:
        pass
    return audit_info
    # trk_items = list(smart_pipeline.find_info_items(info, 'process', 'watch.cli.kwcoco_to_geojson'))
    # pxl_items = list(smart_pipeline.find_info_items(info, 'process', 'watch.tasks.fusion.predict'))
