"""
Test concepts for fsspec wrappers that should make working with S3 / the local
file system seemless.
"""
import ubelt as ub
from watch.utils import util_fsspec


def main_poc():
    local_root = util_fsspec.LocalPath(ub.Path.appdir('watch/tests/ffspec/local_src').ensuredir())
    local_root.delete()
    local_root.ensuredir()

    # We will copy from the remote back to here.
    local_roundtrip_root = util_fsspec.LocalPath(ub.Path.appdir('watch/tests/ffspec/local_roundtrip').ensuredir())

    local_fpath1 = (local_root / 'test_file1.txt')
    local_fpath2 = local_root / 'test_file2.txt'
    local_dpath1 = (local_root / 'dpath1').ensuredir()
    local_fpath3 = (local_dpath1 / 'test_file3.txt')
    local_fpath4 = local_dpath1 / 'test_file4.txt'
    local_fpath1.write_text('data1')
    local_fpath2.write_text('data2')
    local_fpath3.write_text('data3')
    local_fpath4.write_text('data4')

    print("Local FS:")
    # _ = local_root.tree(pathstyle='abs')
    _ = local_root.tree()

    remote_root = util_fsspec.S3Path('s3://kitware-smart-watch-data/testing')

    remote_fpath = remote_root / 'test_remote_file'
    with remote_fpath.open('w') as file:
        file.write('DATA')
    with remote_fpath.open('r') as file:
        print(file.read())

    local_dst_root1 = util_fsspec.LocalPath(ub.Path.appdir('watch/tests/ffspec/dests/local_dst'))
    local_dst_root2 = util_fsspec.LocalPath(ub.Path.appdir('watch/tests/ffspec/dests/nested/local/dst'))

    # Cleanup
    print("Cleanup:")
    remote_root.delete()
    local_dst_root1.delete()
    local_dst_root2.delete()
    local_dst_root1.ensuredir()

    def test_copy(dst_root):
        print("(Before) Destination FS:")
        _ = dst_root.tree()

        dst_dpath1 = dst_root / local_dpath1.relative_to(local_root)
        dst_fpath1 = dst_root / local_fpath1.relative_to(local_root)
        dst_fpath2 = dst_root / local_fpath2.relative_to(local_root)

        local_fpath1.copy(dst_fpath1)
        local_fpath1.copy(dst_fpath1)

        local_fpath1.copy(dst_fpath1)
        local_fpath2.copy(dst_fpath2)

        print(local_root.ls())
        print(dst_root.ls())

        print("(First Dir Copy) Destination FS:")
        local_dpath1.copy(dst_dpath1)
        _ = dst_root.tree()

        print("(Second Dir Copy) Destination FS:")
        local_dpath1.copy(dst_dpath1)
        _ = dst_root.tree()

        tmp_file = (local_dpath1 / 'tmp_new_file')
        tmp_file.write_text('foobar')

        print("(Third Dir Copy (with new file)) Destination FS:")
        local_dpath1.copy(dst_dpath1)
        _ = dst_root.tree()

        print('Copy from dst to dst')
        dst_dpath1.copy(dst_dpath1.parent / 'dpath1_copy')
        dst_dpath1.copy(dst_dpath1.parent / 'dpath1_copy')
        dst_fpath1.copy(dst_fpath1.parent / 'fpath1_copy')
        dst_fpath1.copy(dst_fpath1.parent / 'fpath1_copy')
        _ = dst_root.tree()

        tmp_file.delete()

        print("Copy to the local roundtrip root")
        local_roundtrip_root.tree()

        dst_fpath1.copy(local_roundtrip_root / 'fpath1')
        dst_fpath1.copy(local_roundtrip_root / 'fpath1')

        dst_dpath1.copy(local_roundtrip_root / 'dpath1')
        print("(First Dir Copy) Roundtrip FS:")
        local_roundtrip_root.tree()

        dst_dpath1.copy(local_roundtrip_root / 'dpath1')
        print("(Second Dir Copy) Roundtrip FS:")
        local_roundtrip_root.tree()

    local_roundtrip_root.delete()
    dst_root = remote_root
    test_copy(dst_root)

    local_roundtrip_root.delete()
    dst_root = local_dst_root1
    test_copy(dst_root)

    import pytest
    with pytest.raises(FileNotFoundError):
        # Local copies will fail if the parent directory isn't there
        dst_root = local_dst_root2
        test_copy(dst_root)


def _devcheck():
    import fsspec
    available = fsspec.available_protocols()
    # Local filesystem is "file"
    assert 'file' in available
    assert 'ssh' in available
    assert 's3' in available
    fs = fsspec.filesystem('file')
    fs = fsspec.filesystem('s3')
    fs.ls('s3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval_14_batch_v20/batch/kit/KR_R001/2021-08-31/split/mono/products/kwcoco-dataset/')
    fs.ls('s3://kitware-smart-watch-data/')
    fs.mkdir('s3://kitware-smart-watch-data/testing')

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/poc/fsspec_wrapper_classes.py
    """
    main_poc()
