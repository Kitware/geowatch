"""
Utilities for saving the data that gave rise to particular hash values.
"""
import ubelt as ub
import shelve
import os

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class ReverseHashTable:
    """
    Make a lookup table of hashes we've made, so we can refer to what the heck
    those directory names mean!

    /home/joncrall/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Continue_10GSD_BGR_V003/Drop4_BAS_Continue_10GSD_BGR_V003_epoch=93-step=48128.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_vali.kwcoco/predcfg_1c530993/pred.kwcoco.json

    Example:
        >>> from geowatch.utils.reverse_hashid import *  # NOQA
        >>> data = {'test': 'data'}
        >>> key = ub.hash_data(data)[0:8]
        >>> self = ReverseHashTable(type='test-rhash')
        >>> self.register(key, data)
        >>> self.register('conflict-hash', 'conflict-data1')
        >>> self.register('conflict-hash', 'conflict-data2')
        >>> full_shelf = self.load()
        >>> print('full_shelf = {}'.format(ub.urepr(full_shelf, nl=2)))
    """

    def __init__(self, type='global'):
        from kwutil.util_locks import Superlock
        self.rlut_dpath = ub.Path.appdir('geowatch/hash_rlut', type).ensuredir()
        self.shelf_fpath = self.rlut_dpath / 'hash_rlut.shelf'
        self.text_fpath = self.rlut_dpath / 'hash_rlut.txt'
        self.file_dpath = (self.rlut_dpath / 'hash_rlut').ensuredir()
        self.lock_fpath = self.rlut_dpath / 'flock.lock'
        self.lock = Superlock(thread_key='hash_rlut', lock_fpath=self.lock_fpath)

    def load(self):
        with self.lock:
            shelf = shelve.open(os.fspath(self.shelf_fpath))
            full_shelf = dict(shelf)
        return full_shelf

    def register(self, key, data):
        """
        Args:
            key (str): the hash
            data (Any): the hashed data (must be serializable)
        """
        FULL_TEXT = 1
        DPATH_TEXT = 1

        blake3 = ub.hash_data(data, hasher='blake3')
        row = {'data': data, 'blake3': blake3}
        info = {}
        with self.lock:
            shelf = shelve.open(os.fspath(self.shelf_fpath))
            with shelf:
                # full_shelf = dict(shelf)
                if key not in shelf:
                    datas = shelf[key] = [row]
                    info['status'] = 'new'
                else:
                    datas = shelf[key]
                    found = False
                    for other in datas:
                        if other['blake3'] == row['blake3']:
                            found = True
                            break
                    if not found:
                        info['status'] = 'conflict'
                        datas.append(row)
                        shelf[key] = datas
                    else:
                        info['status'] = 'exists'

                if FULL_TEXT:
                    full_shelf = dict(shelf)
                else:
                    full_shelf = None

            if info['status'] != 'exists':
                # Convinience
                if FULL_TEXT:
                    full_text = ub.urepr(full_shelf, nl=3)
                    self.text_fpath.write_text(full_text)

                if DPATH_TEXT:
                    fpath = self.file_dpath / key
                    datas_text = ub.urepr(datas, nl=3)
                    fpath.write_text(datas_text)
        return info

    @classmethod
    def query(cls, key=None, verbose=1):
        """
        If the type of the hash is unknown, we can search in a few different
        locations for it.
        """
        rlut_root = ub.Path.appdir('geowatch/hash_rlut')
        dpaths = [path for path in rlut_root.iterdir() if path.is_dir()]
        candidates = []
        for dpath in ub.ProgIter(dpaths, desc='rlut is searching', verbose=verbose):
            type = dpath.name
            rlut_type = cls(type)
            full_shelf = rlut_type.load()
            # print('full_shelf = {}'.format(ub.urepr(full_shelf, nl=1, sort=1)))
            if key is None:
                for k, v in full_shelf.items():
                    candidates.append({'found': v, 'type': type, 'key': k})
            elif key in full_shelf:
                candidates.append({'found': full_shelf[key], 'type': type, 'key': key})

        if verbose:
            print(f'Found {len(candidates)} entries for key={key}')
            print('candidates = {}'.format(ub.urepr(candidates, nl=5)))
        return candidates


@profile
def condense_config(params, type, human_opts=None, register=True):
    """
    Given a dictionary of parameters and a type, makes a hash of the params
    prefixes it with a type and ensures it is registered in the global system
    reverse hash lookup table. Some config parts can be given human readable
    descriptions.
    """
    from geowatch.utils.reverse_hashid import ReverseHashTable
    if human_opts is None:
        human_opts = {}
    params = ub.udict(params)
    human_opts = params & human_opts
    other_opts = params - human_opts
    if len(human_opts):
        human_part = ub.urepr(human_opts, compact=1) + '_'
    else:
        human_part = ''
    cfgstr_suffix = human_part + ub.hash_data(other_opts)[0:8]
    cfgstr = f'{type}_{cfgstr_suffix}'
    if register:
        rhash = ReverseHashTable(type=type)
        rhash.register(cfgstr, params)
    return cfgstr
