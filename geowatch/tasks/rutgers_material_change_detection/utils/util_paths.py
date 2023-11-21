import os
import json

from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_repo_dir


def get_dataset_root_dir(dataset_name):
    # Load dataset directory json file.
    base_repo_dir = get_repo_dir()
    dset_dir_json_file_path = os.path.join(base_repo_dir, "transformer", "datasets", "dataset_directories.json")

    ## Check that json file exists.
    if os.path.isfile(dset_dir_json_file_path) is False:
        raise FileNotFoundError(f'Dataset direct json file not found at: "{dset_dir_json_file_path}"')

    dataset_dirs = json.load(open(dset_dir_json_file_path, "r"))

    # Get dataset directory.
    try:
        dset_root_dir = dataset_dirs[dataset_name]
    except KeyError:
        raise KeyError(
            f'Dataset name "{dataset_name}" not found in dataset directory json file at "{dset_dir_json_file_path}"'
        )

    return dset_root_dir


def get_base_paths(key_name):
    # Load base paths json file.
    base_repo_dir = get_repo_dir()
    json_file_path = os.path.join(base_repo_dir, "base_paths.json")

    ## Check that json file exists.
    if os.path.isfile(json_file_path) is False:
        raise FileNotFoundError(f'Base paths json file not found at: "{json_file_path}"')

    base_paths = json.load(open(json_file_path, "r"))

    # Get dataset directory.
    try:
        base_path = base_paths[key_name]
    except KeyError:
        raise KeyError(f'Base path name "{key_name}" not found in json file at "{base_paths}"')

    return base_path
