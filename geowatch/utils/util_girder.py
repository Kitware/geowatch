import ubelt as ub
import os
import fasteners
import threading
from os.path import exists
from os.path import join


PER_PROCESS_THREAD_LOCKS = {}


def grabdata_girder(api_url, resource_id, name=None, dpath=None, hash_prefix=None,
                    appname='geowatch/girder', api_key=None,
                    verbose=1):
    """
    Downloads and caches a file or folder from girder.

    Args:
        api_url (str): the URL to the girder server

        resource_id (str): the id of the resource (e.g. file or folder).

        name (str, default=None): the desired name of the downloaded resource
            on disk if unspecified, the name of the resource on the girder
            server will be used.

        dapth (str, default=None): the path to download the resource to

        hash_prefix (str, default=None):
            If provided, and the item is a file, check that the prefix of the
            hash provided by girder matches this prefix. If the item is a
            folder, then this is not used.

        appname (str, default='geowatch'):
            if ``dpath`` is not given, then the ubelt app_cache_dir for this
            application name is used.

        api_key (str):
            Your API key to autheticate with to access private data.
            If unspecified the ``GIRDER_API_KEY`` environ is used.

        verbose (int, default=1):
            verbosity flag, if non-zero prints download progress.

    Requirements:
        pip install girder-client

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from geowatch.utils.util_girder import *  # NOQA
        >>> from os.path import basename
        >>> api_url = 'https://data.kitware.com/api/v1'
        >>> resource_id = '59eb64678d777f31ac6477eb'
        >>> fpath = grabdata_girder(api_url, resource_id,
        >>>                         hash_prefix='79974f400aa50')
        >>> assert basename(fpath) == 'us.json'
        >>> resource_id = '59eb64298d777f31ac6477e8'
        >>> fpath = grabdata_girder(api_url, resource_id)
        >>> assert basename(fpath) == 'boundries'

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(grabdata_girder))
    """
    # Use the CLI version to get a progress bar
    import girder_client
    if verbose:
        from girder_client.cli import GirderCli
        client = GirderCli(username=None, password=None, apiUrl=api_url)
    else:
        client = girder_client.GirderClient(apiUrl=api_url)

    auth_info = {'api_key': api_key}
    if auth_info.get('api_key', None) is None:
        auth_info['api_key'] = os.environ.get('GIRDER_API_KEY', None)
    if auth_info.get('api_key', None) is not None:
        client.authenticate(apiKey=auth_info['api_key'])

    name = None
    dpath = None
    if dpath is None:
        dpath = ub.Path.appdir(appname).ensuredir()

    get_info_methods = {
        'file': client.getFile,
        'folder': client.getFolder,
        'item': client.getItem,
        'collection': client.getCollection,
    }

    # Determine what type of resource the requested id is for.
    resource_type = None
    for resource_type, get_info in get_info_methods.items():
        try:
            resource_info = get_info(resource_id)
        except girder_client.HttpError as ex:
            if ex.response.status_code == 400:
                pass
            elif ex.response.status_code == 401:
                print('Unauthorized, probably need to set credentials')
                raise
            else:
                print('Unhandled error ex = {}'.format(ex))
            resource_info = None
        except Exception:
            resource_info = None
        else:
            break

    if resource_info is None:
        raise Exception('Unable to determine type of resource')

    if resource_type == 'file':
        file_info = resource_info
        if name is None:
            name = file_info['name']
        dl_path = join(dpath, name)
        dl_method = client.downloadFile
        hash_value = file_info['sha512']
        if hash_prefix:
            if not hash_value.startswith(hash_prefix):
                raise ValueError('Incorrect got={}, want={}'.format(
                    hash_prefix, hash_value))

        depends = file_info['sha512']
    elif resource_type == 'item':
        raise NotImplementedError(
            'The current implementation of downloading items is '
            'not consistent. Download the file instead')
        item_info = resource_info
        dl_method = client.downloadItem
        if name is not None:
            raise ValueError('cannot specify a name for an item')
        else:
            name = item_info['name']
        dl_path = join(dpath, name)
        depends = (
            item_info['updated'],
            item_info['size'],
        )
        if hash_prefix:
            raise ValueError('Cannot specify a hash_prefix for an item')
    elif resource_type == 'folder':
        folder_info = resource_info
        dl_method = client.downloadFolderRecursive
        if name is None:
            name = folder_info['name']
        dl_path = ub.ensuredir((dpath, name))
        depends = (
            folder_info['updated'],
            folder_info['size'],
        )
        if hash_prefix:
            raise ValueError('Cannot specify a hash_prefix for a folder')
    else:
        raise KeyError(resource_type)

    cache_name = resource_type + '_' + name + '.tmp.hash'
    stamp = ub.CacheStamp(cache_name, dpath=dpath, depends=depends)
    if stamp.expired() or not exists(dl_path):
        lock_fpath = join(dpath, cache_name + '.lock')

        # Ensure that multiple processes / threads dont download the same data
        from multiprocessing import current_process
        curr_proc = current_process().name

        # Use a different thread-lock on a per-process / per-stamp basis
        lock_key = (curr_proc, lock_fpath)
        if lock_key not in PER_PROCESS_THREAD_LOCKS:
            PER_PROCESS_THREAD_LOCKS[lock_key] = threading.Lock()
        thread_lock = PER_PROCESS_THREAD_LOCKS[lock_key]

        process_lock = fasteners.InterProcessLock(lock_fpath)  #
        with thread_lock:
            with process_lock:
                if stamp.expired() or not exists(dl_path):
                    dl_method(resource_id, dl_path)
                    stamp.renew()
    return dl_path
