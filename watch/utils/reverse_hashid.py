"""
Utilities for saving the data that gave rise to particular hash values.
"""
import ubelt as ub
import shelve
import os


class ReverseHashTable:
    """
    Make a lookup table of hashes we've made, so we can refer to what the heck
    those directory names mean!

    /home/joncrall/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Continue_10GSD_BGR_V003/Drop4_BAS_Continue_10GSD_BGR_V003_epoch=93-step=48128.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_vali.kwcoco/predcfg_1c530993/pred.kwcoco.json

    Example:
        >>> from watch.utils.reverse_hashid import *  # NOQA
        >>> data = {'test': 'data'}
        >>> key = ub.hash_data(data)[0:8]
        >>> self = ReverseHashTable(type='test-rhash')
        >>> self.register(key, data)
        >>> self.register('conflict-hash', 'conflict-data1')
        >>> self.register('conflict-hash', 'conflict-data2')
        >>> full_shelf = self.load()
        >>> print('full_shelf = {}'.format(ub.repr2(full_shelf, nl=2)))
    """

    def __init__(self, type='global'):
        from watch.utils.util_locks import Superlock
        self.rlut_dpath = ub.Path.appdir('watch/hash_rlut', type).ensuredir()
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
                    full_text = ub.repr2(full_shelf, nl=3)
                    self.text_fpath.write_text(full_text)

                if DPATH_TEXT:
                    fpath = self.file_dpath / key
                    datas_text = ub.repr2(datas, nl=3)
                    fpath.write_text(datas_text)
        return info
