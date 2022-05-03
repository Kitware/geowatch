

def check_long_time():
    import watch
    import ubelt as ub
    import numpy as np
    import kwimage
    import kwarray
    import kwcoco
    import ndsampler
    # dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
    coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_vali.kwcoco.json'
    dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(dset)

    # Grab one video and S2 / L8 images in it
    vidid = list(dset.videos())[0]
    video = dset.index.videos[vidid]
    images = dset.images(vidid=vidid)
    flags = [s != 'WV' for s in images.get('sensor_coarse')]
    images = images.compress(flags)
    gids = list(images)

    # Grab small spatial sample somewhere in the center with a long time extent
    w, h = video['width'], video['height']
    box = kwimage.Boxes([[w, h, 32, 32]], 'cxywh').to_tlbr().quantize()
    space_slices = box.to_slices()[0]
    tr = {
        'vidid': vidid,
        'space_slice': space_slices,
        'gids': gids,
        'channels': 'red|green|blue|nir|swir16|swir22',
    }
    with ub.Timer('load big time small space') as t:
        sample = sampler.load_sample(tr, with_annots=False)
    print(f't.elapsed={t.elapsed}')
    shape = sample['im'].shape
    pixels_per_band = np.prod(shape[0:3])

    spatial_size = 224
    similar_timesteps = int(pixels_per_band / (spatial_size ** 2))
    print(f'similar_timesteps={similar_timesteps}')
    # Grab a large spatial window with a smaller time extent
    w, h = video['width'], video['height']
    box = kwimage.Boxes([[w, h, spatial_size, spatial_size]], 'cxywh').to_tlbr().quantize()
    space_slices = box.to_slices()[0]
    sub_gids = kwarray.shuffle(gids.copy(), rng=12432)[0:similar_timesteps]
    tr = {
        'vidid': vidid,
        'space_slice': space_slices,
        'gids': gids,
        'channels': 'red|green|blue|nir|swir16|swir22',
    }
    with ub.Timer('load small time big space') as t:
        sample = sampler.load_sample(tr, with_annots=False)
    print(f't.elapsed={t.elapsed}')
