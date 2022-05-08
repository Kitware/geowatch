def _annot_pandas_table():
    """
    Notes: it would be nice to construct a pandas table of attributes for each
    annotation.

    This is easy enough using lookup when all properties belong to the
    annotation itself, but for special properties like "cname" or chained
    properties like image.frame_index, it becomes tricker (there is a table
    joint that needs to be resolved).

    These are some notes on how we might implement an API to make this nicer.
    """
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    from kwcoco.coco_objects1d import Categories
    annots = dset.annots()
    annot_props = annots.lookup(['id', 'track_id', 'image_id', 'category_id'])
    image_props = annots.images.unique().lookup(['id', 'frame_index', 'video_id', 'date_captured'])
    cat_props = Categories(annots.lookup('category_id'), dset).unique().lookup(['id', 'name'])

    import pandas as pd
    annot_df = pd.DataFrame(annot_props).set_index('id', drop=True)
    image_df = pd.DataFrame(image_props).set_index('id', drop=True)
    cat_df = pd.DataFrame(cat_props).set_index('id', drop=True).rename({'name': 'cname'}, axis=1)

    merged = annot_df
    merged = merged.join(cat_df, on=['category_id'], how='left')
    merged = merged.join(image_df, on=['image_id'], how='left')

    flags = merged['track_id'].apply(lambda x: x.startswith('BR_'))
    merged = merged[flags]

    import numpy as np
    for track_id, group in merged.groupby('track_id'):
        group = group.sort_values('frame_index')
        is_transition_cent = group['cname'][:-1].values != group['cname'][1:].values
        is_transition_left = np.r_[[True], is_transition_cent]
        is_transition_right = np.r_[is_transition_cent, [True]]
        is_transition = is_transition_left | is_transition_right
        subgroup = group[is_transition]
        if set(group['cname']) == {'No Activity', 'Site Preparation', 'Post Construction', 'Active Construction'}:
            print(subgroup)
