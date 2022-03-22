import ubelt as ub
import os


def find_linked_files(dpath):
    items = []
    prog = ub.ProgIter(desc='walking')
    prog.begin()
    for r, ds, fs in os.walk(dpath):
        r = ub.Path(r)
        prog.update(1)
        for f in fs:
            fpath = r / f
            if fpath.is_symlink():
                real_fpath = fpath.readlink()
                items.append({'fpath': fpath, 'real_fpath': real_fpath})
    prog.end()
    return items


def main():
    """
    We want to see

    (1) if there are symlinks that point to the same data in different datasets
    (2) if there are files that are not referenced by the kwcoco data
    """
    import watch
    dvc_dpath = watch.find_smart_dvc_dpath().augment(suffix='-hdd')
    dpath1 = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15'
    dpath2 = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10'

    linked_files1 = find_linked_files(dpath1)
    linked_files2 = find_linked_files(dpath2)
    for item in linked_files2:
        item['md5'] = item['real_fpath'].name
    for item in linked_files1:
        item['md5'] = item['real_fpath'].name

    md5_to_items1 = ub.group_items(linked_files1, key=lambda x: x['md5'])
    md5_to_items2 = ub.group_items(linked_files2, key=lambda x: x['md5'])

    common = set(md5_to_items1) & set(md5_to_items2)

    for md5 in common:
        items1 = md5_to_items1[md5]
        items2 = md5_to_items2[md5]

    md5_dup1 = {k: v for k, v in ub.map_vals(len, md5_to_items1).items() if v > 1}
    md5_dup2 = {k: v for k, v in ub.map_vals(len, md5_to_items2).items() if v > 1}

    md5_dup1['']

    md5_to_items1['4a67c5b4326903f6ca36a34bd8d531']

    for k in md5_dup2:
        items = md5_to_items2[k]
        print('items = {}'.format(ub.repr2(items, nl=1)))


def find_unregistered_files():
    import watch
    import kwcoco
    dvc_dpath = watch.find_smart_dvc_dpath().augment(suffix='-hdd')
    dpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10'
    coco_fpath = dpath / 'data.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)

    linked_files = find_linked_files(dpath)

    for item in linked_files:
        item['md5'] = item['real_fpath'].name

    md5_to_items = ub.group_items(linked_files, key=lambda x: x['md5'])
    fpath_to_md5 = {x['fpath']: x['md5'] for x in linked_files}

    on_disk_files = {f['fpath'] for f in linked_files}
    in_kwcoco_files = set()
    for coco_img in coco_dset.images().coco_images:
        in_kwcoco_files.update(list(map(ub.Path, coco_img.iter_image_filepaths())))

    # Find all the files that are not registered in the kwcoco data
    print(len(in_kwcoco_files))
    print(len(on_disk_files))
    common = on_disk_files & in_kwcoco_files
    print(len(common))
    unregistered_files = on_disk_files - in_kwcoco_files

    total_unreg_bytes = 0
    for unreg_fpath in ub.ProgIter(unregistered_files):
        total_unreg_bytes += unreg_fpath.lstat().st_size

    import pint
    reg = pint.UnitRegistry()
    (total_unreg_bytes * reg.Unit('bytes')).to('megabytes')

    for unreg_fpath in unregistered_files:
        md5 = fpath_to_md5[unreg_fpath]
        others = md5_to_items[md5]
        if len(others) > 1:
            print('others = {!r}'.format(others))
            print('has others')
        else:
            print('no others')

    total_bytes = 0
    unique_items = []
    for md5, items in ub.ProgIter(md5_to_items.items()):
        item = items[0]
        fpath = item['fpath']
        rel = fpath.relative_to(dpath)
        region, sensor = rel.parts[0:2]
        stat = item['real_fpath'].stat()
        item['st_size'] = stat.st_size
        item['sensor'] = sensor
        item['region'] = region
        unique_items.append(item)
        total_bytes += stat.st_size
    total_gb = (total_bytes * reg.Unit('bytes')).to('terabytes')
    print('total_gb = {!r}'.format(total_gb))

    def bytes_to_gigabytes(x):
        return (x * reg.bytes).to('gigabytes').m

    import pandas as pd
    df = pd.DataFrame(unique_items)
    df.groupby('sensor')['st_size'].sum().apply(bytes_to_gigabytes)
