import os
import subprocess


def get_file_from_s3(uri, path):
    dst_path = os.path.join(path, os.path.basename(uri))
    try:
        subprocess.check_call(
            ['aws', 's3', 'cp', '--quiet', uri, dst_path]
        )
    except Exception:
        raise OSError('Error getting file from s3URI: ' + uri)

    return dst_path


def send_file_to_s3(path, uri):
    try:
        subprocess.check_call(
            ['aws', 's3', 'cp', '--quiet', path, uri]
        )
    except Exception:
        raise OSError('Error sending file to s3URI: ' + uri)
    return uri
