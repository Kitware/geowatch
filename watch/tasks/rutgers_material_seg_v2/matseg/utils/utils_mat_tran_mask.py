import numpy as np
from collections import defaultdict


def _compute_natural_to_manmade_material_transform_heuristic_old(early_mat_pred, late_mat_pred):
    # Compute heuristic rules for material transition (natural to unnatural).
    n_x, n_y = np.where((early_mat_pred == 1) | (early_mat_pred == 2) | (early_mat_pred == 4) |
                        (early_mat_pred == 5))
    un_x, un_y = np.where((late_mat_pred == 3) | (late_mat_pred == 6) | (late_mat_pred == 7) |
                          (late_mat_pred == 8))
    natural_mask = np.zeros(early_mat_pred.shape)
    natural_mask[n_x, n_y] = 1

    unnatural_mask = np.zeros(early_mat_pred.shape)
    unnatural_mask[un_x, un_y] = 1

    x, y = np.where((natural_mask == 1) & (unnatural_mask == 1))
    mtm = np.zeros(early_mat_pred.shape)
    mtm[x, y] = 1

    return mtm


def _compute_natural_to_manmade_material_transform_heuristic(early_mat_pred,
                                                             late_mat_pred,
                                                             heuristic='soften_seasonal'):
    # Compute heuristic rules for material transition (natural to unnatural).
    ## Create mappings.
    mat_mappings = []
    if heuristic == 'basic':
        mat_mappings.append([[4, 2], 1])  # veg -> soil
        mat_mappings.append([[4, 7], 1])  # veg -> asphalt
        mat_mappings.append([[4, 6], 1])  # veg -> polymer
        mat_mappings.append([[5, 2], 1])  # snow -> soil
        mat_mappings.append([[5, 7], 1])  # snow -> asphalt
        mat_mappings.append([[5, 6], 1])  # snow -> polymer
        mat_mappings.append([[1, 2], 1])  # water -> soil
        mat_mappings.append([[1, 7], 1])  # water -> asphalt
        mat_mappings.append([[1, 6], 1])  # water -> polymer
        mat_mappings.append([[7, 6], 1])  # asphalt -> polymer
        mat_mappings.append([[6, 7], 1])  # polymer -> asphalt
        mat_mappings.append([[2, 7], 1])  # soil -> asphalt
        mat_mappings.append([[2, 6], 1])  # soil -> polymer
    elif heuristic == 'soften_seasonal':
        mat_mappings.append([[4, 2], 0.2])  # veg -> soil
        mat_mappings.append([[4, 7], 1])  # veg -> asphalt
        mat_mappings.append([[4, 6], 1])  # veg -> polymer
        mat_mappings.append([[5, 2], 1])  # snow -> soil
        mat_mappings.append([[5, 7], 1])  # snow -> asphalt
        mat_mappings.append([[5, 6], 1])  # snow -> polymer
        mat_mappings.append([[1, 2], 0.2])  # water -> soil
        mat_mappings.append([[1, 7], 1])  # water -> asphalt
        mat_mappings.append([[1, 6], 1])  # water -> polymer
        mat_mappings.append([[7, 6], 1])  # asphalt -> polymer
        mat_mappings.append([[6, 7], 1])  # polymer -> asphalt
        mat_mappings.append([[2, 7], 1])  # soil -> asphalt
        mat_mappings.append([[2, 6], 1])  # soil -> polymer
    else:
        raise NotImplementedError(f'Heuristic {heuristic} not implemented.')

    mapping_conditions = defaultdict(list)
    for mat_map, mat_value in mat_mappings:
        early_mat, late_mat = mat_map
        mapping_conditions[mat_value].append((early_mat_pred == early_mat) &
                                             (late_mat_pred == late_mat))

    mtm = np.zeros(early_mat_pred.shape)
    for mat_value, mapping_conditions in mapping_conditions.items():
        x, y = np.where(np.logical_or.reduce(mapping_conditions))
        mtm[x, y] = mat_value

    return mtm


def compute_material_transition_mask(mode,
                                     first_frames,
                                     last_frames,
                                     first_quality_frames=None,
                                     last_quality_frames=None,
                                     heuristic='basic'):
    """Compute material transition masks (MTM) based on mode setting.

    Args:
        mode (str): The method for computing the MTM.
        first_frames (np.array): _description_ [n_frames, n_classes, height, width]
        last_frames (np.array): _description_ [n_frames, n_classes, height, width]
        first_quality_frames (np.array): A binary array of shape [n_frames, height, width] the same
            resolution as the frames and the same number of masks.
        last_quality_frames (np.array): A binary array of shape [n_frames, height, width] the same
            resolution as the frames and the same number of masks.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
        _type_: _description_
        _type_: _description_
    """
    # natural_mat_ids = [1, 2, 4, 5]
    # unnatural_mat_ids = [3, 6, 7, 8]

    if mode == 'hard_class_1':
        # Average X first and final frame confidences.
        early_mat_pred = first_frames.sum(axis=0).argmax(axis=0)
        late_mat_pred = last_frames.sum(axis=0).argmax(axis=0)

        # Compute heuristic rules for material transition (natural to unnatural).
        mtm = _compute_natural_to_manmade_material_transform_heuristic(early_mat_pred,
                                                                       late_mat_pred,
                                                                       heuristic=heuristic)
        # n_x, n_y = np.where((avg_beg_mat_pred == 1) | (avg_beg_mat_pred == 2) |
        #                     (avg_beg_mat_pred == 4) | (avg_beg_mat_pred == 5))
        # un_x, un_y = np.where((avg_end_mat_pred == 3) | (avg_end_mat_pred == 6) |
        #                       (avg_end_mat_pred == 7) | (avg_end_mat_pred == 8))
        # natural_mask = np.zeros(avg_beg_mat_pred.shape)
        # natural_mask[n_x, n_y] = 1

        # unnatural_mask = np.zeros(avg_beg_mat_pred.shape)
        # unnatural_mask[un_x, un_y] = 1

        # x, y = np.where((natural_mask == 1) & (unnatural_mask == 1))
        # mat_trans_mask = np.zeros(avg_beg_mat_pred.shape)
        # mat_trans_mask[x, y] = 1

    elif mode == 'hard_class_2':
        # Note: Same as hard_class_2 but ignore any snow classes.

        # Set snow probabilities to zero.
        beg_mat_confs = first_frames.sum(axis=0)
        end_mat_confs = last_frames.sum(axis=0)
        beg_mat_confs[5, :, :] = 0
        end_mat_confs[5, :, :] = 0

        # Average X first and final frame confidences.
        early_mat_pred = beg_mat_confs.argmax(axis=0)
        late_mat_pred = end_mat_confs.argmax(axis=0)

        # Compute heuristic rules for material transition (natural to unnatural).
        # n_x, n_y = np.where((avg_beg_mat_pred == 1) | (avg_beg_mat_pred == 2) |
        #                     (avg_beg_mat_pred == 4) | (avg_beg_mat_pred == 5))
        # un_x, un_y = np.where((avg_end_mat_pred == 3) | (avg_end_mat_pred == 7) |
        #                       (avg_end_mat_pred == 8))
        # natural_mask = np.zeros(avg_beg_mat_pred.shape)
        # natural_mask[n_x, n_y] = 1

        # unnatural_mask = np.zeros(avg_beg_mat_pred.shape)
        # unnatural_mask[un_x, un_y] = 1

        # x, y = np.where((natural_mask == 1) & (unnatural_mask == 1))
        # mat_trans_mask = np.zeros(avg_beg_mat_pred.shape)
        # mat_trans_mask[x, y] = 1
        mtm = _compute_natural_to_manmade_material_transform_heuristic(early_mat_pred,
                                                                       late_mat_pred,
                                                                       heuristic=heuristic)

    elif mode == 'hard_quality_1':
        if (first_quality_frames is None) or (last_quality_frames is None):
            raise ValueError(f'{mode}: Requires quality masks to compute properly.')

        # Make sure quality masks have allow all pixels to have valid values.
        fx, fy = np.where(first_quality_frames.sum(axis=0) == 0)
        lx, ly = np.where(last_quality_frames.sum(axis=0) == 0)

        n_first_invalid_pixels = fx.shape[0]
        n_last_invalid_pixels = lx.shape[0]
        if (n_first_invalid_pixels != 0) or (n_last_invalid_pixels != 0):
            raise ValueError(
                f'Quality mask does not contain any valid values for {n_first_invalid_pixels} first pixels and {n_last_invalid_pixels} last pixels.'
            )

        # Sum material confidence for valid pixels.
        agg_first_frames = np.multiply(first_frames, first_quality_frames[:, None]).sum(axis=0)
        agg_last_frames = np.multiply(last_frames, last_quality_frames[:, None]).sum(axis=0)

        # Get materials with highest probabilities.
        early_mat_pred = agg_first_frames.argmax(axis=0)
        late_mat_pred = agg_last_frames.argmax(axis=0)

        # Compute material transition mask.
        mtm = _compute_natural_to_manmade_material_transform_heuristic(early_mat_pred,
                                                                       late_mat_pred,
                                                                       heuristic=heuristic)

    elif mode == 'hard_quality_2':
        if (first_quality_frames is None) or (last_quality_frames is None):
            raise ValueError(f'{mode}: Requires quality masks to compute properly.')

        # Make sure quality masks have allow all pixels to have valid values.
        fx, fy = np.where(first_quality_frames.sum(axis=0) == 0)
        lx, ly = np.where(last_quality_frames.sum(axis=0) == 0)

        # Sum material confidence for valid pixels.
        agg_first_frames = np.multiply(first_frames, first_quality_frames[:, None]).sum(axis=0)
        agg_last_frames = np.multiply(last_frames, last_quality_frames[:, None]).sum(axis=0)

        # Get materials with highest probabilities.
        early_mat_pred = agg_first_frames.argmax(axis=0)
        late_mat_pred = agg_last_frames.argmax(axis=0)

        # Compute material transition mask.
        mtm = _compute_natural_to_manmade_material_transform_heuristic(early_mat_pred,
                                                                       late_mat_pred,
                                                                       heuristic=heuristic)

        mtm[fx, fy] = 0
        mtm[lx, ly] = 0

    else:
        raise NotImplementedError(f'Mode "{mode}" not implemented.')

    return mtm, early_mat_pred, late_mat_pred


if __name__ == '__main__':
    # TEST 1: Make sure hard_quality_1 runs properly.
    mode = 'hard_quality_1'
    n_frames, n_classes, height, width = 5, 8, 200, 200
    first_frames = np.zeros([n_frames, n_classes, height, width], dtype=float)
    last_frames = np.zeros([n_frames, n_classes, height, width], dtype=float)
    first_quality = np.ones([n_frames, height, width], dtype='uint8')
    last_quality = np.ones([n_frames, height, width], dtype='uint8')

    mtm = compute_material_transition_mask(mode,
                                           first_frames,
                                           last_frames,
                                           first_quality_frames=first_quality,
                                           last_quality_frames=last_quality)
    print('Test 1: PASSED')

    # TEST 2: Make sure hard_quality_1 fails on missing quality masks.
    first_quality = None
    try:
        mtm = compute_material_transition_mask(mode,
                                               first_frames,
                                               last_frames,
                                               first_quality_frames=first_quality,
                                               last_quality_frames=last_quality)
    except ValueError:
        print('Test 2: PASSED')

    # TEST 3: Make sure hard_quality_1 fails on no quality values.
    first_quality = np.zeros([n_frames, height, width], dtype='uint8')
    try:
        mtm = compute_material_transition_mask(mode,
                                               first_frames,
                                               last_frames,
                                               first_quality_frames=first_quality,
                                               last_quality_frames=last_quality)
    except ValueError:
        print('Test 3: PASSED')
