"""
SeeAlso:

"""
import kwcoco
import ubelt as ub
import zipfile
import pandas as pd
import xdev


class DMJ_Paths:
    def __init__(self):
        self.dmj_dpath = ub.Path('/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan')

    def bundle_dpath(self, region_id):
        return self.dmj_dpath / region_id / 'kwcoco-dataset'

    def sensor_dpath(self, region_id, sensor):
        return self.dmj_dpath / region_id / 'kwcoco-dataset' / region_id / sensor

    def coco_fpath(self, region_id):
        return self.dmj_dpath / region_id / 'kwcoco-dataset' / 'cropped_kwcoco.json'

    def all_region_ids(self):
        region_ids = [p.parent.name for p in list(self.dmj_dpath.glob('*/kwcoco-dataset'))]
        return region_ids


class HDD_Paths:
    def __init__(self):
        import watch
        self.hdd_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
        self.hdd_bundle_dpath = self.hdd_data_dpath / 'Drop6'
        self.geojson_annot_dpath = self.hdd_data_dpath / 'annotations/drop6'

    def sensor_zip_fpath(self, region_id, sensor):
        return self.hdd_bundle_dpath / region_id / (sensor + '.zip')

    def imgonly_coco_fpath(self, region_id):
        return self.hdd_bundle_dpath / f'imgonly-{region_id}.kwcoco.json'

    def imganns_coco_fpath(self, region_id):
        return self.hdd_bundle_dpath / f'imganns-{region_id}.kwcoco.zip'

    def all_imgonly_fpaths(self):
        return list(self.hdd_bundle_dpath.glob('imgonly*.kwcoco.*'))

    def all_region_ids(self):
        region_ids = []
        for p in self.all_imgonly_fpaths():
            region_id = p.name.split('.')[0].split('-')[-1]
            region_ids.append(region_id)
        return region_ids

    def all_zip_fpaths(self):
        existing_zip_fpaths = list(HDD.hdd_bundle_dpath.glob('*/*.zip'))
        return existing_zip_fpaths


DMJ = DMJ_Paths()
HDD = HDD_Paths()


def check_for_new_stuff():
    # from watch.utils import util_time
    # from datetime import datetime as datetime_cls
    # import xdev
    import watch
    dmj_dpath = DMJ.dmj_dpath

    hdd_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
    hdd_bundle_dpath = hdd_data_dpath / 'Drop6'

    sorted(dmj_dpath.ls(), key=lambda x: x.stat().st_mtime)

    new_regions = [
        'AE_C001', 'AE_C002', 'AE_C003', 'BH_R001', 'BR_R001',
        'BR_R002', 'BR_R004', 'CH_R001', 'KR_R001', 'KR_R002',
        'LT_R001', 'NZ_R001', 'PE_C003', 'PE_R001', 'QA_C001',
        'SA_C005', 'US_C000', 'US_C010', 'US_C011', 'US_C012',
        'US_C014', 'US_R001', 'US_R004', 'US_R005', 'US_R006',
        'US_R007'
        'PE_C001',
    ]

    # disagree_regions = [
    #     'PE_C001',
    #     'LT_R001',
    # ]

    # TODO: how do we auto-determine this?
    new_regions = ['AE_C001', 'AE_C002', 'PE_R001', 'US_R004', 'US_R006', 'PE_C001', 'US_R007', 'CH_R001', 'US_C011']

    # new_regions = ['AE_C001', 'PE_R001']
    zip_region_assets(new_regions)

    dvc_add_new_regions(new_regions)

    copy_dmj_kwcoco_files(new_regions)

    reprocess_all_kwcoco_files(hdd_data_dpath, hdd_bundle_dpath)

    from watch.utils import simple_dvc
    splits_fpath = HDD.hdd_bundle_dpath / 'splits.zip'
    dvc = simple_dvc.SimpleDVC.coerce(splits_fpath)
    dvc.add(splits_fpath, verbose=1)
    dvc.git_commitpush(message='update images and annots')
    dvc.push(splits_fpath, remote='aws', verbose=1)


def copy_dmj_kwcoco_files(new_regions):
    comparisons = []
    for region_id in ub.ProgIter(new_regions):
        comparison = compare_kwcoco_vs_dmj(region_id)
        comparisons.append(comparison)

    status_to_comparisons = ub.udict(ub.group_items(comparisons, key=lambda x: x['status']))
    status_to_comparisons.map_values(len)

    to_copy = status_to_comparisons.get('different-needs_copy', [])
    to_copy += status_to_comparisons.get('missing_jpc-needs_copy', [])

    for comparison in ub.ProgIter(to_copy):
        comparison['dmj_coco_fpath'].copy(comparison['imgonly_fpath'], overwrite=True)


def all_checks_for_region(region_id):
    """
    Do an exuastive check for a single region. To see what is out of sync.

        region_id = 'LT_R001'
        region_id = 'PE_C001'
    """
    kwcoco_compare = compare_kwcoco_vs_dmj(region_id)
    print('kwcoco_compare = {}'.format(ub.urepr(kwcoco_compare, nl=1)))

    hdd_self_compare = list(check_hdd_annotations_agree_with_zipfiles(region_id))
    dmj_self_compare = list(check_dmj_annotation_agree_with_disk(region_id))
    print('hdd_self_compare = {}'.format(ub.urepr(hdd_self_compare, nl=1)))
    print('dmj_self_compare = {}'.format(ub.urepr(dmj_self_compare, nl=1)))

    sensor_to_compare = {}
    for sensor in ['S2', 'L8', 'WV', 'WV1', 'PD']:
        zipfile_compare = compare_zipfile_vs_dmj(region_id, sensor)
        print('zipfile_compare = {}'.format(ub.urepr(zipfile_compare, nl=1)))
        sensor_to_compare[sensor] = zipfile_compare


def compare_kwcoco_vs_dmj(region_id):
    """
    Check to see if the kwcoco files point to the same assets.
    Other aspects may be different.

    Ignore:
        region_id = 'LT_R001'
        compare_kwcoco_vs_dmj(region_id)

        region_id = 'PE_C001'
        compare_kwcoco_vs_dmj(region_id)
    """
    import xdev
    dmj_coco_fpath = DMJ.coco_fpath(region_id)
    imgonly_fpath = HDD.imgonly_coco_fpath(region_id)
    comparison = {}
    comparison['region_id'] = region_id
    comparison['imgonly_fpath'] = imgonly_fpath
    comparison['dmj_coco_fpath'] = dmj_coco_fpath

    if not dmj_coco_fpath.exists():
        comparison['status'] = 'missing-dmj'
    elif not imgonly_fpath.exists():
        comparison['status'] = 'missing_jpc-needs_copy'
    else:
        import kwcoco
        dmj_dset = kwcoco.CocoDataset(dmj_coco_fpath)
        jpc_dset = kwcoco.CocoDataset(imgonly_fpath)

        dmj_assets = []
        jpc_assets = []
        for c in dmj_dset.images().coco_images:
            dmj_assets.extend([o['file_name'] for o in c.iter_asset_objs()])

        for c in jpc_dset.images().coco_images:
            jpc_assets.extend([o['file_name'] for o in c.iter_asset_objs()])

        jpc_assets = sorted(jpc_assets)
        dmj_assets = sorted(dmj_assets)
        overlaps = xdev.set_overlaps(jpc_assets, dmj_assets, s1='jpc', s2='dmj')
        comparison['overlaps'] = overlaps
        if dmj_assets == jpc_assets:
            comparison['status'] = 'same'
        else:
            comparison['status'] = 'different-needs_copy'
    return comparison


def compare_zipfile_vs_dmj(region_id, sensor):
    """
    Check that the data in the zipfile is the same as the data in the DMJ
    assets for a region / sensor.

    Ignore:
        region_id = 'US_C011'
        sensor = 'WV'
        region_id = 'PE_C001'
        sensor = 'WV'
    """
    import xdev
    import zipfile
    comparison = {}
    jpc_zip_fpath = HDD.sensor_zip_fpath(region_id, sensor)
    dmj_sensor_dpath = DMJ.sensor_dpath(region_id, sensor)

    if not jpc_zip_fpath.exists() and not dmj_sensor_dpath.exists():
        comparison['status'] = 'both-dont-exist'
        return comparison
    elif not jpc_zip_fpath.exists():
        comparison['status'] = 'jpc-does-not-exist'
        return comparison
    elif not dmj_sensor_dpath.exists():
        comparison['status'] = 'dmj-does-not-exist'
        return comparison

    jpc_zfile = zipfile.ZipFile(jpc_zip_fpath)
    zipped_names = jpc_zfile.namelist()
    zip_assets = [ub.Path(n) for n in zipped_names if not n.endswith('/') and '.tmp' not in str(n)]
    dmj_assets = [
        p.relative_to(dmj_sensor_dpath.parent)
        for p in dmj_sensor_dpath.rglob('*') if p.is_file()
    ]
    if 1:
        # sensor = dmj_sensor_dpath.name
        region_id = dmj_sensor_dpath.parent.name
        new = []
        for p in zip_assets:
            if p.parts[0] == region_id:
                p = ub.Path(*p.parts[1:])
            if '.tmp' not in str(p):
                new.append(p)
        zip_assets = new
    comparison['overlaps'] = xdev.set_overlaps(zip_assets, dmj_assets, s1='jpc', s2='dmj')
    if comparison['overlaps']['isect'] == comparison['overlaps']['union']:
        comparison['status'] = 'same'
    else:
        comparison['status'] = 'different'
    return comparison


def cleanup_tmpfiles_in_zips():
    for zip_fpath in HDD.all_zip_fpaths():
        ...
        zfile = zipfile.ZipFile(zip_fpath)
        zipped_names = zfile.namelist()
        zip_assets = [ub.Path(n) for n in zipped_names if not n.endswith('/')]
        bad_assets = [p  for p in zip_assets if '.tmp' in str(p)]

        total_size1 = 0
        total_size2 = 0

        for part in bad_assets:
            total_size1 += zfile.NameToInfo[str(part)].file_size
            total_size2 += zfile.NameToInfo[str(part)].compress_size

        xdev.byte_str(total_size1)
        xdev.byte_str(total_size2)

        if bad_assets:
            print(f'zip_fpath={zip_fpath}')
            raise Exception


def check_dmj_annotation_agree_with_disk(region_id):
    coco_fpath = DMJ.coco_fpath(region_id)
    dset = kwcoco.CocoDataset(coco_fpath)
    # assert not dset.missing_images(check_aux=True)

    assets = ub.ddict(list)
    for coco_img in dset.images().coco_images:
        sensor = coco_img.img['sensor_coarse']
        assets[sensor].extend([ub.Path(o['file_name']) for o in coco_img.iter_asset_objs()])

    for sensor, assets in assets.items():
        sensor_dpath = DMJ.sensor_dpath(region_id, sensor)
        if sensor_dpath.exists():
            disk_assets = [p.relative_to(dset.bundle_dpath) for p in list(sensor_dpath.rglob('*.tif'))]
            new = []
            for p in disk_assets:
                if '.tmp' not in str(p):
                    new.append(p)
            disk_assets = new
        else:
            disk_assets = []
        overlaps = xdev.set_overlaps(disk_assets, assets, s1='dmj-disk', s2='dmj-coco')
        is_same = overlaps['isect'] == overlaps['union']
        comparison = {
            'sensor': sensor,
            'region_id': region_id,
            'overlaps': overlaps,
            'is_same': is_same
        }
        yield comparison


def check_hdd_annotations_agree_with_zipfiles(region_id):
    imgonly_fpath = HDD.imganns_coco_fpath(region_id)
    dset = kwcoco.CocoDataset(imgonly_fpath)

    jpc_assets = ub.ddict(list)
    for coco_img in dset.images().coco_images:
        sensor = coco_img.img['sensor_coarse']
        jpc_assets[sensor].extend([ub.Path(o['file_name']) for o in coco_img.iter_asset_objs()])

    for sensor, assets in jpc_assets.items():
        sensor_zip_fpath = HDD.sensor_zip_fpath(region_id, sensor)
        if sensor_zip_fpath.exists():
            zfile = zipfile.ZipFile(sensor_zip_fpath)
            zipped_names = zfile.namelist()
            zip_assets = [ub.Path(n) for n in zipped_names if not n.endswith('/')]
            new = []
            for p in zip_assets:
                if p.parts[0] != region_id:
                    p = ub.Path(region_id, *p.parts)
                if '.tmp' not in str(p):
                    new.append(p)
            zip_assets = new
        else:
            zip_assets = []
        overlaps = xdev.set_overlaps(zip_assets, assets, s1='asset-zip', s2='asset-coco')
        is_same = overlaps['isect'] == overlaps['union']
        comparison = {
            'sensor': sensor,
            'region_id': region_id,
            'overlaps': overlaps,
            'is_same': is_same
        }

        set(assets) - set(zip_assets)
        set(zip_assets) - set(assets)
        yield comparison


def all_check_dmj_annotation_agree_with_self():
    """
    Check that the annotations in DMJ bundles agree with the images there.
    """
    region_ids = DMJ.all_region_ids()
    issues = []
    comparisons = []
    for region_id in ub.ProgIter(region_ids):
        for comparison in check_dmj_annotation_agree_with_disk(region_id):
            comparisons.append(comparison)
            if not comparison['is_same']:
                issues.append(comparison)

    print('issues = {}'.format(ub.urepr(issues, nl=1)))

    dmj_regions = DMJ.all_region_ids()
    issue_regions = [x['region_id'] for x in issues]
    set(dmj_regions) & set(issue_regions)


def all_check_hdd_annotations_agree_with_zipfiles():
    """
    Check that the annotations agree with zipped images in the HDD bundle.
    """
    region_ids = HDD.all_region_ids()
    issues = []
    comparisons = []
    for region_id in ub.ProgIter(region_ids):
        for comparison in check_hdd_annotations_agree_with_zipfiles(region_id):
            comparisons.append(comparison)
            if not comparisons['is_same']:
                issues.append(comparison)

    print('issues = {}'.format(ub.urepr(issues, nl=1)))
    dmj_regions = DMJ.all_region_ids()
    issue_regions = [x['region_id'] for x in issues]
    set(dmj_regions) & set(issue_regions)


def dvc_add_new_regions(new_regions):
    SLOW_CHECK = 0
    import rich
    needs_add = []
    for region_id in ub.ProgIter(new_regions, verbose=3):
        for sensor in ['S2', 'L8', 'WV1', 'PD', 'WV']:
            sensor_zip_fpath_dst = HDD.sensor_zip_fpath(region_id, sensor)
            if sensor_zip_fpath_dst.exists():
                if sensor_zip_fpath_dst.is_symlink():
                    continue
                sidecar_fpath = sensor_zip_fpath_dst.augment(tail='.dvc')
                if not sidecar_fpath.exists():
                    rich.print('[red] does not exist, need to add')
                    needs_add.append(sensor_zip_fpath_dst)
                else:
                    from watch.utils import util_yaml
                    data = util_yaml.Yaml.loads(sidecar_fpath.read_text())

                    if SLOW_CHECK:
                        print('check hash')
                        md5_tracker = data['outs'][0]['md5']
                        print(f'md5_tracker={md5_tracker}')
                        md5_got = ub.hash_file(sensor_zip_fpath_dst, hasher='md5')
                        print(f'md5_got={md5_got}')
                        from watch.utils import util_yaml
                        sidecar_fpath = ub.Path(sidecar_fpath)
                        if md5_got == md5_tracker:
                            rich.print('[green] all good')
                        else:
                            needs_add.append(sensor_zip_fpath_dst)
                            rich.print('[red] md5 mismatch, need to add')
                    else:
                        needs_add.append(sensor_zip_fpath_dst)
                        rich.print('[blue] maybe add?')

    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce(needs_add[0])
    dvc.add(needs_add, verbose=1)
    # dvc.push(needs_add, verbose=2, remove='aws')


def compare_all_zipfiles_vs_dmj():
    """
     Inspect: US_C011 / WV
     Inspect: US_R007 / WV
     Inspect: US_R007 / L8
     Inspect: US_R007 / S2
     Inspect: CH_R001 / WV
     Inspect: CH_R001 / L8
     Inspect: CH_R001 / S2

     Inspect: LT_R001 / WV
     57/88... rate=1.25 Hz, eta=0:00:24, total=0:00:45
     Inspect: LT_R001 / L8
     58/88... rate=1.25 Hz, eta=0:00:23, total=0:00:46
     Inspect: LT_R001 / S2

    """
    import rich
    existing_zip_fpaths = list(HDD.hdd_bundle_dpath.glob('*/*.zip'))
    tocheck = []
    for jpc_zip_fpath in ub.ProgIter(existing_zip_fpaths, verbose=3):
        sensor = jpc_zip_fpath.name.split('.')[0]
        region_id = jpc_zip_fpath.parent.name
        tocheck.append({'region_id': region_id, 'sensor': sensor})

    if 0:
        tocheck = [
            {'region_id': 'LT_R001', 'sensor': 'S2'},
            {'region_id': 'LT_R001', 'sensor': 'L8'},
            {'region_id': 'LT_R001', 'sensor': 'WV'},
        ]

    for row in ub.ProgIter(tocheck, verbose=3):
        sensor = row['sensor']
        region_id = row['region_id']
        jpc_zip_fpath = HDD.sensor_zip_fpath(region_id, sensor)
        dmj_sensor_dpath = DMJ.sensor_dpath(region_id, sensor)
        if dmj_sensor_dpath.exists():
            overlaps = compare_zipfile_vs_dmj(region_id, sensor)
            if overlaps['isect'] != overlaps['union']:
                rich.print(f'[red] Inspect: {region_id} / {sensor} - {overlaps}')


def check_missing_image_against_dmj():
    import watch
    ssd_bundle_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd') / 'Drop6'
    hdd_bundle_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd') / 'Drop6'

    fpath = ssd_bundle_dpath / 'data.kwcoco.zip'
    dset = kwcoco.CocoDataset(fpath)
    missing = dset.missing_images(check_aux=True)

    rows = []
    region_sensor_to_missing = ub.ddict(list)
    for i, p, g in missing:
        region_id, sensor = ub.Path(p).parts[0:2]
        rows.append({
            'region_id': region_id,
            'sensor': sensor,
        })
        region_sensor_to_missing[(region_id, sensor)].append(p)
    df = pd.DataFrame(rows)
    counts = df.value_counts()
    print(counts)

    for vidid in dset.videos():
        first_img = dset.images(video_id=vidid).coco_images[0]
        print(first_img.video['name'])
        print(first_img.resolution(space='video'))

    # from watch.utils import util_time
    import watch
    drop6_geojson_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd') / 'annotations/drop6'

    region_groups = counts.reset_index().groupby('region_id')
    fixup_tasks = []

    for region_id, group in region_groups:
        issues = []
        dmj_coco_fpath = DMJ.coco_fpath(region_id)
        # Note: this check may not work anymore, we reprocessed the images
        comparison = compare_kwcoco_vs_dmj(region_id)
        if comparison['status'] == 'same':
            issues.append(f'We didnt copy the {region_id} kwcoco file')
            jpc_img_coco_fpath = hdd_bundle_dpath / f'imgonly-{region_id}.kwcoco.json'
            jpc_ann_coco_fpath = hdd_bundle_dpath / f'imganns-{region_id}.kwcoco.zip'
            fixup_tasks.append({
                'type': 'copy',
                'dmj_fpath': dmj_coco_fpath,
                'imgonly_fpath': jpc_img_coco_fpath,
                'imganns_fpath': jpc_ann_coco_fpath,
                'region_id': region_id,
            })

        for sensor in group['sensor']:
            dmj_sensor_dpath = DMJ.sensor_dpath(region_id, sensor)
            jpc_zip_fpath = hdd_bundle_dpath / region_id / (sensor + '.zip')
            if dmj_sensor_dpath.exists():
                assert jpc_zip_fpath.exists()
                comparison = compare_zipfile_vs_dmj(region_id, sensor)
                if comparison['status'] == 'same':
                    issues.append(
                        f'assets in zipfile {comparison} are different than dmj for {sensor}'
                    )
            # row_missing = region_sensor_to_missing[(region_id, sensor)]
            # for p in row_missing:
            #     (dmj_bundle_dpath / p).exists()

        print(f'region_id={region_id}')
        print('issues = {}'.format(ub.urepr(issues, nl=1)))

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=5)

    for task in fixup_tasks:
        if task['type'] == 'copy':
            region_id = task['region_id']
            copy_job = queue.submit(f'''cp {task['dmj_fpath']} {task['imgonly_fpath']}''')
            queue.submit(ub.codeblock(
                fr'''
                python -m watch reproject_annotations \
                    --src "{task['imgonly_fpath']}" \
                    --dst "{task['imganns_fpath']}" \
                    --propogate_strategy="SMART" \
                    --site_models="{drop6_geojson_dpath}/site_models/{region_id}_*" \
                    --region_models="{drop6_geojson_dpath}/region_models/{region_id}*"
                '''), depends=copy_job)
    queue.rprint()
    queue.run()

    command = ub.codeblock(
        fr'''
        python -m watch.cli.prepare_splits \
            --base_fpath="{hdd_bundle_dpath}/imganns*.kwcoco.*" \
            --workers=5 \
            --constructive_mode=True --run=1
        ''')
    print(command)

    # Move annotations over to the ssd
    for hdd_fpath in ub.ProgIter(list(hdd_bundle_dpath.glob('*.kwcoco.*'))):
        ssd_fpath = ssd_bundle_dpath / hdd_fpath.name
        hdd_fpath.copy(ssd_fpath, overwrite=True)

    # Commit them
    command = ub.codeblock(
        r'''
        rm -rf splits.zip
        7z a splits.zip -mx9 -- *.kwcoco.*
        dvc add splits.zip
        git commit -am "Update annotations"
        git push
        dvc push -r aws splits.zip
        ''')
    print(command)


def remove_bad_zip_paths():
    dpath = ub.Path('.')
    bad_zip_paths = []
    for zip_fpath in dpath.glob('*/*.zip'):
        zfile = zipfile.ZipFile(zip_fpath)
        if len(zfile.namelist()) == 0:
            bad_zip_paths.append(zip_fpath)
        zfile = None

    for zip_fpath in bad_zip_paths:
        dvc_fpath = zip_fpath.augment(tail='.dvc')
        assert dvc_fpath.exists()
        dvc_fpath.delete()
        zip_fpath.delete()

    for fpath in dpath.glob('imganns*.kwcoco.*'):
        if 'BR_R001' in fpath.name:
            continue
        dset = kwcoco.CocoDataset(fpath)
        missing = dset.missing_images(check_aux=True)
        if missing:
            raise Exception


def reprocess_all_kwcoco_files():
    import cmd_queue
    geojson_annot_dpath = HDD.geojson_annot_dpath
    imgonly_fpaths = list(HDD.hdd_bundle_dpath.glob('imgonly*.kwcoco.*'))

    rows = []
    for imgonly_fpath in imgonly_fpaths:
        region_id = imgonly_fpath.name.split('.')[0].split('-')[1]
        rows.append({
            'region_id': region_id,
            'imgonly_fpath': imgonly_fpath,
        })

    queue = cmd_queue.Queue.create(backend='tmux', size=15)
    for row in rows:
        imgonly_fpath = row['imgonly_fpath']
        command = ub.codeblock(
            rf'''
            python -m watch.cli.coco_add_watch_fields \
                --src {imgonly_fpath} \
                --dst {imgonly_fpath} \
                --target_gsd 10
            ''')
        queue.submit(command)
    queue.run()

    # US_R006, US_R007

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=15)
    for row in rows:
        imgonly_fpath = row['imgonly_fpath']
        region_id = row['region_id']
        imganns_fpath = HDD.imganns_coco_fpath(region_id)
        assert imgonly_fpath.exists()
        command = ub.codeblock(
            fr'''
            python -m watch reproject_annotations \
                --src "{imgonly_fpath}" \
                --dst "{imganns_fpath}" \
                --propogate_strategy="SMART" \
                --site_models="{geojson_annot_dpath}/site_models/{region_id}_*" \
                --region_models="{geojson_annot_dpath}/region_models/{region_id}*"
            ''')
        print(command)
        queue.submit(command)
    queue.rprint()
    queue.run()

    command = ub.codeblock(
        fr'''
        python -m watch.cli.prepare_splits \
            --base_fpath="{HDD.hdd_bundle_dpath}/imganns*.kwcoco.zip" \
            --workers=5 \
            --constructive_mode=True --run=1
        ''')
    ub.cmd(command, system=True)

    command = ub.codeblock(
        fr'''
        python -m watch.cli.prepare_splits \
            --base_fpath="{HDD.hdd_bundle_dpath}/imganns*.kwcoco.zip" \
            --workers=5 \
            --constructive_mode=True --run=1
        ''')
    ub.cmd(command, system=True)

    # Repackage splits
    splits_fpath = HDD.hdd_bundle_dpath / 'splits.zip'
    splits_fpath.delete()

    import xdev
    with xdev.ChDir(HDD.hdd_bundle_dpath):
        ub.cmd(f'7z a {splits_fpath} -- *.kwcoco.*', system=True, cwd=HDD.hdd_bundle_dpath)

    # # Move annotations over to the ssd
    # ssd_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
    # ssd_bundle_dpath = ssd_data_dpath / 'Drop6'
    # for hdd_fpath in ub.ProgIter(list(hdd_bundle_dpath.glob('*.kwcoco.*'))):
    #     ssd_fpath = ssd_bundle_dpath / hdd_fpath.name
    #     hdd_fpath.copy(ssd_fpath, overwrite=True)

    # # Commit them
    # command = ub.codeblock(
    #     r'''
    #     rm -rf splits.zip
    #     7z a splits.zip -mx9 -- *.kwcoco.*
    #     dvc add splits.zip
    #     git commit -am "Update annotations"
    #     git pull
    #     git push
    #     dvc push -r aws splits.zip
    #     ''')
    # print(command)


def zip_region_assets(new_regions):
    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=15)

    for region_id in new_regions:
        dmj_bundle = DMJ.bundle_dpath(region_id)
        dmj_assets = dmj_bundle / region_id
        for dpath in dmj_assets.ls():
            if dpath.is_dir():
                sensor_dpath = dpath
                sensor = dpath.name
                sensor_zip_fpath_dst = HDD.sensor_zip_fpath(region_id, sensor)
                # Add command to zip up all of the data for the sensor into the
                # DVC directory.
                deljob = queue.submit(f'rm -f {sensor_zip_fpath_dst}')
                command = f'7z a -mx=0 {sensor_zip_fpath_dst} {sensor_dpath}'
                queue.submit(command, depends=deljob)
    queue.rprint()
    queue.run()
