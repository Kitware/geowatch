#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoShardConfig(scfg.Config):
    """
    Shards a kwcoco dataset into multiple subparts
    """
    default = {
        'src': scfg.Value(None, help='input dataset to split', position=1),

        'dst_pattern': scfg.Value('{dpath}/shard_{stem}_{index:03d}.kwcoco.json', help='pattern for writing a shard'),

        'num_shards': scfg.Value(None, help='number of shards'),

        'max_shard_size': scfg.Value(None, help='maximum size of each shard'),

        'rng': scfg.Value(None, help='random seed'),
    }


def main(cmdline=True, **kw):
    """
    Example:
        >>> from watch.cli.coco_shard import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('shapes7')
        >>> kw = {
        >>>     'src': dset.fpath,
        >>>     'max_shard_size': 2,
        >>> }
        >>> cmdline = False
        >>> main(cmdline, **kw)
    """
    import kwcoco
    config = CocoShardConfig(kw, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    if config['src'] is None:
        raise Exception('must specify source: {}'.format(config['src']))

    dst_pattern = config['dst_pattern']
    num_shards = config['num_shards']
    max_shard_size = config['max_shard_size']

    print('reading fpath = {!r}'.format(config['src']))
    dset = kwcoco.CocoDataset.coerce(config['src'])
    gids = dset.images().gids

    gid_chunks = list(ub.chunks(gids, nchunks=num_shards,
                                chunksize=max_shard_size))

    src_fpath = ub.Path(dset.fpath)

    stem = src_fpath.stem.split('.')[0]
    dpath = src_fpath.parent

    shard_infos = []
    for index, gids in enumerate(gid_chunks):
        fmtdict = {
            'dpath': dpath,
            'index': index,
            'stem': stem,
        }
        dst_fpath = dst_pattern.format(**fmtdict)
        info = {
            'dst_fpath': dst_fpath,
            'gids': gids,
        }
        shard_infos.append(info)

    dest_fpaths = [d['dst_fpath'] for d in shard_infos]
    duplicates = ub.find_duplicates(dest_fpaths)
    if duplicates:
        raise ValueError('Destination shard filenames: {}'.format(duplicates))

    for info in ub.ProgIter(shard_infos, desc='sharding'):
        gids = info['gids']
        dset_shard = dset.subset(gids)
        dset_shard.fpath = dst_fpath
        dset_shard.dump(dset_shard.fpath, newlines=True)
    print('Wrote shards: {}'.format(ub.repr2(dest_fpaths, nl=1)))

_CLI = CocoShardConfig

if __name__ == '__main__':
    _CLI.main()
