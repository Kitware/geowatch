import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from geowatch.tasks.rutgers_material_change_detection.utils.util_paths import get_dataset_root_dir

from geowatch.tasks.rutgers_material_change_detection.datasets.iarpa_kwdataset import IARPA_KWDATASET
from geowatch.tasks.rutgers_material_change_detection.datasets.peo_dataset import PassiveEarthObservationDataset

dataset_classes = {
    "iarpa_drop1": IARPA_KWDATASET,
    "iarpa_drop2": IARPA_KWDATASET,
    "iarpa_drop1_ta1": IARPA_KWDATASET,
    "peo": PassiveEarthObservationDataset,
}


def custom_collate_fn(
    data,
    stack_keys=[
        "video",
        "total_bin_change",
        "total_sem_change",
        "pw_bin_change",
        "pw_sem_change",
        "seq_frame_pred",
        "active_frames",
        "anchor",
        "positive",
        "negative",
        "ss_arrow_of_time",
        "ss_splice_change",
        "ss_splice_change_index",
        "ss_mat_recon",
        "sem_seg",
    ],
):
    out_data = {}
    example = data[0]
    for key in example.keys():
        if key in stack_keys:
            # Stack items like default collate function.
            out = []
            for ex in data:
                out.append(ex[key])
            out = torch.stack(out, 0)
        else:
            # Return a list of items.
            out = []
            for ex in data:
                out.append(ex[key])
        out_data[key] = out
    return out_data


def build_dataset(
    dset_name,
    split,
    video_slice,
    task_mode,
    transforms=None,
    seed_num=0,
    normalize_mode=None,
    channels=None,
    max_iterations=None,
    overwrite_dset_dir=None,
):
    if overwrite_dset_dir:
        root_dir = overwrite_dset_dir
    else:
        root_dir = get_dataset_root_dir(dset_name)

    try:
        dataset = dataset_classes[dset_name](
            root_dir,
            split,
            video_slice,
            task_mode,
            transforms=transforms,
            seed_num=seed_num,
            normalize_mode=normalize_mode,
            channels=channels,
            max_iterations=max_iterations,
        )
    except KeyError:
        raise NotImplementedError(f"Build dataset call not implemented for: {dset_name}")

    return dataset


def compute_dataset_example_weights(dataset, weight_method="ratio_pct"):
    """[summary]

    Args:
        dataset ([type]): [description]
        weight_method (str, optional): [description]. Defaults to 'ratio_pct'.

    Returns:
        list: A probabilty for each example in dataset.

    Raises:
        NotImplementedError: [description]
    """
    assert (
        dataset.task_mode == "total_bin_change"
    ), f'Weight sampler not implemented for non total_bin_change tasks such as "{dataset.task_mode}".'

    if weight_method == "ratio_pct":

        crop_ratios = []
        for i in range(len(dataset)):
            example = dataset.__getitem__(i)
            target = example[dataset.task_mode].numpy()

            # TODO: Update for non binary labels
            x0, _ = np.where(target == 0)
            x1, _ = np.where(target > 0)

            # Compute ratio of annotations to blank regions.
            n_blank_pixels = x0.shape[0]
            n_anno_pixels = x1.shape[0]

            n_total_viable_pixels = n_blank_pixels + n_anno_pixels

            if n_total_viable_pixels == 0:
                continue

            ratio = n_anno_pixels / n_total_viable_pixels
            crop_ratios.append(ratio)

        # Normalize to probability distribution.
        crop_ratios = np.asarray(crop_ratios)
        offset_val = crop_ratios[np.nonzero(crop_ratios)[0]].min()
        if offset_val == 0:
            offset_val += 0.001
        crop_ratios += offset_val
        crop_ratios /= crop_ratios.sum()

        percentages = list(crop_ratios)

    elif weight_method == "ratio_split":
        ratio_split_val = 0.75

        crop_ratios = []
        for i in range(len(dataset)):
            example = dataset.__getitem__(i)
            target = example[dataset.task_mode].numpy()

            # TODO: Update for non binary labels
            x0, _ = np.where(target == 0)
            x1, _ = np.where(target > 0)

            # Compute ratio of annotations to blank regions.
            n_blank_pixels = x0.shape[0]
            n_anno_pixels = x1.shape[0]

            n_total_viable_pixels = n_blank_pixels + n_anno_pixels

            if n_total_viable_pixels == 0:
                continue

            ratio = n_anno_pixels / n_total_viable_pixels
            crop_ratios.append(ratio)

        # Normalize to probability distribution.
        crop_ratios = np.asarray(crop_ratios)
        split_value = np.quantile(crop_ratios, ratio_split_val)

        percentages = np.zeros(crop_ratios.shape[0])

        upper_indices = np.where(crop_ratios >= split_value)[0]
        lower_indices = np.where(crop_ratios < split_value)[0]

        n_upper_examples = upper_indices.shape[0]
        n_lower_examples = lower_indices.shape[0]

        assert (n_upper_examples + n_lower_examples) == crop_ratios.shape[0]

        upper_pct = 0.5 / n_upper_examples
        lower_pct = 0.5 / n_lower_examples

        percentages[upper_indices] = upper_pct
        percentages[lower_indices] = lower_pct

        percentages = list(percentages)

    elif weight_method == "pos_neg":
        # Find if examples have contain non-zero labels.
        crop_binary = []
        for i in range(len(dataset)):
            example = dataset.__getitem__(i)
            target = example[dataset.task_mode].numpy()

            # Get all non-zero or non-ignored labels.
            x1, _ = np.where(target > 0)
            n_anno_pixels = x1.shape[0]

            if n_anno_pixels > 0:
                crop_binary.append(1)
            else:
                crop_binary.append(0)

        # Normalize to probability distribution.
        crop_binary = np.asarray(crop_binary)

        # Get positive and negative example indices.
        percentages = np.zeros(crop_binary.shape[0])
        pos_indices = np.where(crop_binary == 1)[0]
        neg_indices = np.where(crop_binary == 0)[0]

        n_pos_examples = pos_indices.shape[0]
        n_neg_examples = neg_indices.shape[0]

        assert (n_pos_examples + n_neg_examples) == crop_binary.shape[0]

        # Compute normalized percentages for positive and negative samples.
        pos_pct = 0.5 / n_pos_examples
        neg_pct = 0.5 / n_neg_examples

        # Populate example percentage probability with probability depending on positive or negative example.
        percentages[pos_indices] = pos_pct
        percentages[neg_indices] = neg_pct

        percentages = list(percentages)

    else:
        raise NotImplementedError(f'Sampling method "{weight_method}" not implemented.')

    return percentages


def create_loader(dataset, split, batch_size, n_workers, collate_fn=custom_collate_fn, example_sampler_method=None):
    # Get example sampler.
    if example_sampler_method is not None:
        print("Generating sampling weights for each example.")
        ex_weights = compute_dataset_example_weights(dataset, example_sampler_method)
        sampler = WeightedRandomSampler(ex_weights, len(ex_weights), replacement=True)
    else:
        sampler = None

    if split == "train":
        if sampler is not None:
            shuffle = False
        else:
            shuffle = True

        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=sampler,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    return loader
