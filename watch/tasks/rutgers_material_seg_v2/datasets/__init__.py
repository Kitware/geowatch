import torch
from datetime import datetime

from watch.tasks.rutgers_material_seg_v2.datasets.image_dataset import ImageDataset
from watch.tasks.rutgers_material_seg_v2.datasets.material_pixel_dataset import MaterialPixelDataset

DATASETS = {'material_pixel': {'training': MaterialPixelDataset, 'evaluation': ImageDataset}}


def build_dataset(dataset_name, dataset_path, split, **kwargs):
    if split in ['train', 'valid']:
        split_type = 'training'
    elif split in ['test', 'eval']:
        split_type = 'evaluation'
    else:
        raise NotImplementedError

    try:
        dataset_type = DATASETS[dataset_name]
    except KeyError:
        raise NotImplementedError(f'Dataset "{dataset_name}" has not been implemented in the build_dataset function.')

    try:
        dataset_func = dataset_type[split_type]
    except KeyError:
        raise NotImplementedError(
            f'The {split_type} dataset for "{dataset_name}" has not been implemented for build_dataset function.')

    dataset = dataset_func(dataset_path, split, **kwargs)

    return dataset


def custom_collate_fn(dset_outputs):
    out_data = {}
    keys = dset_outputs[0].keys()
    for key in keys:
        # Return a list of items.
        out = []
        for output in dset_outputs:
            if output[key] is None:
                pass

            if type(output[key]) == datetime:
                out.append(output[key])
            else:
                out.append(torch.tensor(output[key]))

        # Handle edge case variables from output of dataset class.
        if type(dset_outputs[0][key]) == datetime:
            pass
        else:
            try:
                out = torch.stack(out, 0)
            except:
                breakpoint()
                pass
        out_data[key] = out

    return out_data