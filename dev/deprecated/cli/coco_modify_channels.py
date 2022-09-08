"""
Rename or normalize channel names

TODO:
    - [ ] Currently only hacked to work with S2 images, but should extend later
"""
import kwcoco
# import ubelt as ub
import scriptconfig as scfg


class CocoModifyChannelConfig(scfg.Config):
    """
    Combine kwcoco files with different "auxiliary" features into a single
    kwcoco file.

    TODO:
        - [ ] This might go in kwcoco proper? This could be folded into "union"
    """
    default = {
        'src': scfg.Value([], help='path to dataset', position=1),

        'dst': scfg.Value(None, help='dataset to write to'),

        'normalize': scfg.Value(False, help='if True, use the standard normalizations'),

        'modify': scfg.Value(None, help='search and replacements are colon separted, multiple channels are comma separated. E.g. find1:replace1,find2:replace2'),
    }


def main(cmdline=False, **kwargs):
    """
    from os.path import join
    dvc_dpath = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/')
    kwargs = {
        'src': join(dvc_dpath, 'extern/onera_2018/onera_train.kwcoco.json')
    }


    """
    config = CocoModifyChannelConfig(default=kwargs, cmdline=cmdline)
    fpath = config['src']
    dset = kwcoco.CocoDataset.coerce(fpath)

    # Set the filepath to the one we will write to
    dset.fpath = config['dst']

    if config['modify'] is not None:
        raise NotImplementedError

    if config['normalize']:
        from watch.utils import util_bands
        S2_normalizer = {
            info['name']: info['common_name']
            for info in util_bands.SENTINEL2
            if 'common_name' in info
        }

        for img in dset.index.imgs.values():
            auxiliary = img.get('auxiliary', [])
            if len(auxiliary) == 13:
                hueristic = 'S2'
            else:
                raise NotImplementedError

            if hueristic == 'S2':
                for aux in auxiliary:
                    # TODO: make more robust
                    orig_chan_code = aux['channels']
                    normed = S2_normalizer.get(orig_chan_code, orig_chan_code)
                    aux['channels'] = normed

    dset.dump(dset.fpath, newlines=True)
    return dset


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.coco_modify_channels --help
    """
    main(cmdline=True)
