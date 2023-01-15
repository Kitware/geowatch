"""
In the interest of development speed we can't always be perfect about correctly
passing the right metadata to functions that need it. This module serves as a
place to store hard-coded heuristics so we are explicit about where we are
cutting corners or using magic numbers. The idea is that  this will make it
easier for us to go back and make the code robust.
"""
import ubelt as ub


# # FIXME: Hard-coded category aliases.
# https://smartgitlab.com/TE/standards
# # The correct way to handle these would be to have some information in the
# # kwcoco category dictionary that specifies how the categories should be
# # interpreted.
# _HEURISTIC_CATEGORIES = {
#     'background': {'background', 'No Activity', 'Post Construction'},
#     'pre_background': {'No Activity'},
#     'post_background': {'Post Construction'},
#     'ignore': {'ignore', 'Unknown', 'clouds'},
# }


# TODO: ensure consistency with IARPA?
# https://smartgitlab.com/TE/annotations/-/wikis/Annotation-Status-Types
# https://smartgitlab.com/TE/metrics-and-test-framework/-/blob/main/iarpa_smart_metrics/evaluation.py#L1205
# NOTE: A "Status" is not a category.
# It indicates what sort of annotation detail is available.
HUERISTIC_STATUS_DATA = [
    {'name': 'seen', 'color': 'cyan'},
    {'name': 'train', 'color': 'cyan'},

    {'name': 'ignore', 'color': 'lightsalmon'},

    # Note: 'colors for these status labels are undefined, using neutral gray
    {'name': 'negative', 'color': 'gray'},
    {'name': 'negative_unbounded', 'color': 'gray'},

    {'name': 'positive_excluded', 'color': 'gray'},

    {'name': 'positive_annotated', 'color': 'black'},
    {'name': 'positive_annotated_static', 'color': 'black'},
    {'name': 'positive_partial', 'color': 'black'},
    {'name': 'positive_pending', 'color': 'black'},

    {'name': 'positive_unbounded', 'color': 'darkviolet'},

    # TODO? Add alias of pending for "positive_pending"? For QFabric?

    {'name': 'system_confirmed', 'color': 'kitware_blue'},
]


# "Positive Match Confusion" is the label the truth is given when it has some
# match in our set of positive predictions.  Denote what type of confusion a
# truth status incurs when it is matched.
PHASE_STATUS_TO_MATCHED_CONFUSION = {
    'seen'                      : 'gt_seen',
    'train'                     : 'gt_seen',

    'ignore'                    : 'gt_ignore',

    'negative'                  : 'gt_false_pos',
    'negative_unbounded'        : 'gt_false_pos',

    'positive_excluded'         : 'gt_false_pos',

    'positive_annotated'        : 'gt_true_pos',
    'positive_annotated_static' : 'gt_true_pos',
    'positive_partial'          : 'gt_true_pos',
    'positive_pending'          : 'gt_true_pos',

    'positive_unbounded'        : 'gt_positive_unbounded',

}

# Mapping of annotation status to the kwcoco category name
# Used in project annotations
PHASE_STATUS_TO_KWCOCO_CATNAME = {
    'seen'                     : None,
    'train'                    : None,

    'ignore'                    : 'ignore',

    'negative'                  : 'negative',
    'negative_unbounded'        : 'negative',

    'positive_annotated'        : None,  # This must have a category already do not map
    'positive_annotated_static' : None,  # This must have a category already do not map
    'positive_excluded'         : 'ignore',    # This is positive, but is not "big" enough
    'positive_partial'          : 'positive',  # Does not have phase labels
    'positive_pending'          : 'positive',  # Does not have phase labels

    'positive_unbounded'        : 'positive',  # Start or end date might not be defined
}

IARPA_STATUS_TO_INFO = {row['name']: row for row in HUERISTIC_STATUS_DATA}

# update HUERISTIC_STATUS_DATA
for name, row in IARPA_STATUS_TO_INFO.items():
    if name in PHASE_STATUS_TO_KWCOCO_CATNAME:
        row['kwcoco_catname'] = PHASE_STATUS_TO_KWCOCO_CATNAME[name]

for name, row in IARPA_STATUS_TO_INFO.items():
    if name in PHASE_STATUS_TO_MATCHED_CONFUSION:
        row['positive_match_confusion'] = PHASE_STATUS_TO_MATCHED_CONFUSION[name]


if 0:
    import pandas as pd
    print(pd.DataFrame(HUERISTIC_STATUS_DATA).to_string())


IARPA_REAL_STATUS = {
    'positive': ["positive_annotated", "positive_annotated_static", "positive_partial", "positive_pending"],
    'negative': ["positive_excluded", "negative", "negative_unbounded"],
    'ignore': 'ignore',
}


IARPA_CONFUSION_COLORS = {}
IARPA_CONFUSION_COLORS['gt_true_neg'] = 'darkgreen'  # no IARPA color for this, make one up.
IARPA_CONFUSION_COLORS['gt_true_pos'] = 'lime'
IARPA_CONFUSION_COLORS['gt_false_pos'] = 'red'
IARPA_CONFUSION_COLORS['gt_false_neg'] = 'black'
IARPA_CONFUSION_COLORS['gt_positive_unbounded'] = "darkviolet"
IARPA_CONFUSION_COLORS['gt_ignore'] = "lightsalmon"
IARPA_CONFUSION_COLORS['gt_seen'] = "gray"
IARPA_CONFUSION_COLORS['sm_pos_match'] = "orange"
IARPA_CONFUSION_COLORS['sm_partially_wrong'] = "aquamarine"
IARPA_CONFUSION_COLORS['sm_completely_wrong'] = "magenta"


def iarpa_assign_truth_confusion(truth_status, has_positive_match):
    """
    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> import pandas as pd
        >>> rows = []
        >>> for truth_status in IARPA_STATUS_TO_INFO.keys():
        >>>     for has_positive_match in [0, 1]:
        >>>         gt_cfsn = iarpa_assign_truth_confusion(truth_status, has_positive_match)
        >>>         rows.append({
        >>>             'truth_status': truth_status,
        >>>             'has_positive_match': has_positive_match,
        >>>             'confusion': gt_cfsn,
        >>>         })
        >>> print(pd.DataFrame(rows).to_string())
    """
    gt_cfsn = None
    if has_positive_match:
        gt_cfsn = IARPA_STATUS_TO_INFO[truth_status].get('positive_match_confusion', None)
    else:
        if truth_status in ["positive_unbounded"]:
            gt_cfsn = 'gt_positive_unbounded'
        elif truth_status in ["ignore"]:
            gt_cfsn = 'gt_ignore'
        elif truth_status in ['seen', 'train']:
            gt_cfsn = 'gt_seen'
        elif truth_status in IARPA_REAL_STATUS['negative']:
            gt_cfsn = 'gt_true_neg'
        elif truth_status in IARPA_REAL_STATUS['positive']:
            gt_cfsn = 'gt_false_neg'

    return gt_cfsn


def iarpa_assign_pred_confusion(truth_match_statuses):
    """
    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> import itertools as it
        >>> truth_match_statuses = {'positive_partial', 'positive_excluded'}
        >>> for combo in it.combinations(IARPA_STATUS_TO_INFO, 2):
        >>>     truth_match_statuses = combo
        >>>     pred_cfsn = iarpa_assign_pred_confusion(truth_match_statuses)
        >>>     print(f'{pred_cfsn=} for {truth_match_statuses}')
    """
    pred_cfsn = None
    if not truth_match_statuses:
        pred_cfsn = 'sm_completely_wrong'

    truth_cfsns = {
        IARPA_STATUS_TO_INFO[s].get('positive_match_confusion', None)
        for s in truth_match_statuses
    }
    if 'gt_true_pos' in truth_cfsns:
        if 'gt_false_pos' in truth_cfsns:
            pred_cfsn = 'sm_partially_wrong'
        else:
            pred_cfsn = 'sm_pos_match'
    elif 'gt_false_pos' in truth_cfsns:
        pred_cfsn = 'sm_completely_wrong'

    return pred_cfsn


# metrics-and-test-framework/evaluation.py:1684
# Note: the condition field is in development the idea is to
# encode when a category should be added as a label.

def TAG_IF(tag, condition):
    return {'type': 'condition', 'op': 'tag_if', 'tag': tag, 'condition': condition}


def CONDITION(op, args):
    return {'type': 'condition', 'op': op, 'args': args}


def ALL(*args):
    return {'type': 'condition', 'op': 'all', 'args': args}


CATEGORIES = [
    # TODO: clouds
    {
        'name': 'background',
        'color': 'black',
        'scored': False,
        'required': True,
        'tags': ['background']
    },

    # 'color' lightsalmon
    {
        'name': 'ignore',
        'color': 'violet',
        'scored': False,
        'required': True,
        'tags': ['ignore']
    },

    # {'name': 'clouds', 'color': 'offwhite', 'scored': False},
    {
        'name': 'Unknown',
        'color': 'blueviolet',
        'scored': False,
        'tags': ['ignore'],
    },

    {
        'name': 'positive',
        'color': 'palegreen',
        'scored': False,
        # 'conditional_tags': [
        #     TAG_IF('positive', CONDITION('TASK_EQ', 'saliency')),
        #     TAG_IF('ignore', CONDITION('TASK_EQ', 'class')),
        # ],
    },

    {
        'name': 'negative',
        'color': 'gray',
        'scored': False,
        'tags': ['background'],
        # 'conditional_tags': [
        #     TAG_IF('hard_negative', CONDITION('TASK_EQ', 'class')),
        # ],
    },

    {
        'name': 'Site Preparation',
        'color': 'gold',
        'scored': True,
        'tags': ['positive'],
    },
    {
        'name': 'Active Construction',
        'color': 'lime',
        'scored': True,
        'tags': ['positive'],
    },
    # Conditional classes
    {
        'name': 'Post Construction',
        'color': 'darkturquoise',
        'scored': True,
        'tags': ['positive'],
        # 'conditional_tags': [
        #     TAG_IF('background', CONDITION('TASK_EQ', 'saliency')),
        #     # Only positive if task=CLASS and has context
        #     TAG_IF('positive', ALL(
        #         CONDITION('TASK_EQ', 'class'),
        #         CONDITION('ALSO_HAS', [
        #             'Site Preparation', 'Active Construction', 'No Activity'],
        #         )
        #     )),
        # ],
    },
    {
        'name': 'No Activity',
        'color': 'tomato',
        'scored': True,
        'tags': ['saliency'],
        # 'conditional_tags': [
        #     TAG_IF('background', CONDITION('TASK_EQ', 'saliency')),
        #     # Only positive if task=CLASS and has context
        #     TAG_IF('positive', ALL(
        #         CONDITION('TASK_EQ', 'class'),
        #         CONDITION('ALSO_HAS', [
        #             'Site Preparation', 'Active Construction', 'No Activity'],
        #         )
        #     )),
        # ],
    },
]


def hack_track_categories(track_catnames, task):
    """
    Returns:
        List[str]: Modified categories

    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> basis = {
        >>>     #'task': ['class', 'saliency'],
        >>>     'task': ['class'],
        >>>     'track_catnames': [
        >>>         ['No Activity'],
        >>>         ['Post Construction'],
        >>>         ['Post Construction', 'Post Construction', ],
        >>>         ['No Activity', 'ignore', 'ignore'],
        >>>         ['No Activity', 'Post Construction'],
        >>>         ['No Activity', 'Site Preparation', 'Post Construction'],
        >>>     ],
        >>> }
        >>> for kw in ub.named_product(basis):
        >>>     task = kw['task']
        >>>     track_catnames = kw['track_catnames']
        >>>     print('kw = {}'.format(ub.repr2(kw, nl=1)))
        >>>     print(hack_track_categories(track_catnames, task))

    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> track_catnames = ['negative', 'negative']
        >>> task = 'saliency'
        >>> result = hack_track_categories(track_catnames, task)
        >>> print(result)
        ['negative', 'negative']

    """
    # FIXME! This is hard coded nonsense, need to come up with a general
    # way to encode these conditions in the categories themselves. Getting
    # this right is harder than I want it to be, so I'm hacking it.

    # Might want to make a real parser for this mini-language, or find an
    # existing mini-language that works

    main_classes = {'No Activity', 'Site Preparation', 'Post Construction', 'Post Construction'}

    # This is some of the uggliest code I've ever written
    new_catnames = []
    if task == 'saliency':
        for catname in track_catnames:
            if catname == 'No Activity':
                catname = 'background'
            elif catname == 'Post Construction':
                catname = 'background'
            elif catname == 'Unknown':
                catname = 'ignore'
            new_catnames.append(catname)
    elif task == 'class':
        unique_catnames = set(track_catnames)
        for catname in track_catnames:
            if catname == 'No Activity':
                remain = unique_catnames - {catname, None}
                if len(remain & main_classes) > 0:
                    catname = catname
                elif len(remain) == 0:
                    catname = 'background'
                elif remain.issubset({'background', 'ignore', 'Unknown'}):
                    catname = 'ignore'
            elif catname == 'Post Construction':
                remain = unique_catnames - {catname, None}
                if len(remain & main_classes) > 0:
                    catname = catname
                elif len(remain) == 0:
                    catname = 'background'
                elif remain.issubset({'background', 'ignore'}):
                    catname = 'ignore'
            elif catname == 'positive':
                catname = 'ignore'
            elif catname == 'negative':
                catname = 'background'
            elif catname == 'Unknown':
                catname = 'ignore'
            new_catnames.append(catname)
    else:
        raise KeyError(task)
        # op = condition['op']
        # # TODO: normalize classes
        # # TODO: make label conditionals as part of kwcoco
        # if op == 'ALSO_HAS':
        #     track_catnames = set(track_catnames)
        #     flag = any(arg in track_catnames for arg in condition['args'])
        #     return flag
        # else:
        #     raise NotImplementedError(op)
    return new_catnames


# Backwards compat (remove if nothing uses them)
CATEGORIES_SCORED = [c for c in CATEGORIES if c.get('scored', False)]
CATEGORIES_UNSCORED = [c for c in CATEGORIES if not c.get('scored', False)]

# CATEGORIES_UNSCORED = [
#     {'name': 'positive', 'color': 'olive', 'scored': False},
# ]
# CATEGORIES = CATEGORIES_SCORED + CATEGORIES_UNSCORED


# Might need to split this up into a finer-grained structure
# IGNORE_CLASSNAMES = {'clouds', 'occluded', 'ignore', 'unknown', 'Unknown'}
# BACKGROUND_CLASSES = {c['name'] for c in CATEGORIES if 'background' in c.get('tags', {})}
# UNDISTINGUISHED_CLASSES =  {c['name'] for c in CATEGORIES if 'saliency' in c.get('tags', {})}
IGNORE_CLASSNAMES = {'ignore', 'Unknown'}
BACKGROUND_CLASSES = {'background'}
NEGATIVE_CLASSES = {'negative'}
UNDISTINGUISHED_CLASSES =  {'positive'}
CONTEXT_CLASSES = {'No Activity', 'Post Construction'}
# 'background',
# 'No Activity',
# 'Post Construction',
# 'negative',
# }
# SPECIAL_CONTEXT_CLASSES = {
#     'No Activity',
#     'Post Construction',
#     'negative',
# }


# # These classes are used in BAS, but not in AC/SC
# UNDISTINGUISHED_CLASSES = {
#     'positive',
# }


CATEGORIES_DCT = {
    'positive': {
        'scored': [
            {'name': 'Site Preparation', 'color': 'gold'},
            {'name': 'Active Construction', 'color': 'lime'},
            {'name': 'Post Construction', 'color': 'darkturquoise'},

        ],
        'unscored': [
            {'name': 'positive', 'color': 'green'},
        ],
    },
    'negative': {
        'scored': [
            # Maybe this should not be marked as "scored", because it isn't
            {'name': 'No Activity', 'color': 'tomato'},
        ],
        'unscored': [
            {'name': 'Unknown', 'color': 'blueviolet'},
            {'name': 'ignore', 'color': 'slategray'},
            {'name': 'negative', 'color': 'orangered'},
            {'name': 'background', 'color': 'black'},
        ],
    }
}

# 'name' field only
CNAMES_DCT = {
    k1: {k2: [cat['name'] for cat in cats]
         for k2, cats in dct.items()}
    for k1, dct in CATEGORIES_DCT.items()
}


# For passing site summaries from BAS to SC
SITE_SUMMARY_CNAME = 'Site Boundary'


def ensure_heuristic_coco_colors(coco_dset, force=False):
    """
    Args:
        coco_dset (kwcoco.CocoDataset): object to modify
        force (bool): if True, overwrites existing colors if needed

    TODO:
        - [ ] Move this non-heuristic functionality to
            :func:`kwcoco.CocoDataset.ensure_class_colors`

    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo()
        >>> ensure_heuristic_coco_colors(coco_dset)
        >>> assert all(c['color'] for c in coco_dset.cats.values())
    """
    for hcat in CATEGORIES:
        cat = coco_dset.index.name_to_cat.get(hcat['name'], None)
        if cat is not None:
            if force or cat.get('color', None) is None:
                cat['color'] = hcat['color']
    data_dicts = coco_dset.dataset['categories']
    _ensure_distinct_dict_colors(data_dicts)


def ensure_heuristic_category_tree_colors(classes, force=False):
    """
    Args:
        classes (kwcoco.CategoryTree): object to modify
        force (bool): if True, overwrites existing colors if needed

    TODO:
        - [ ] Move this non-heuristic functionality to
            :func:`kwcoco.CategoryTree.ensure_colors`

    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> import kwcoco
        >>> classes = kwcoco.CategoryTree.coerce(['ignore', 'positive', 'Active Construction', 'foobar', 'Unknown', 'baz'])
        >>> ensure_heuristic_category_tree_colors(classes)
        >>> assert all(d['color'] for n, d in classes.graph.nodes(data=True))
    """
    # Set any missing class color with the heuristic category
    for hcat in CATEGORIES:
        node_data = classes.graph.nodes.get(hcat['name'], None)
        if node_data is not None:
            if force or node_data.get('color', None) is None:
                node_data['color'] = hcat['color']
    data_dicts = [data for node, data in classes.graph.nodes(data=True)]
    _ensure_distinct_dict_colors(data_dicts)


def _ensure_distinct_dict_colors(data_dicts, force=False):
    # Generalized part that could move to kwcoco
    have_dicts = [d for d in data_dicts if d.get('color', None) is not None]
    miss_dicts = [d for d in data_dicts if d.get('color', None) is None]
    num_uncolored = len(miss_dicts)
    if num_uncolored:
        import kwimage
        existing_colors = [kwimage.Color(d['color']).as01() for d in have_dicts]
        new_colors = kwimage.Color.distinct(
            num_uncolored, existing=existing_colors, legacy=False)
        for d, c in zip(miss_dicts, new_colors):
            d['color'] = c


HUERISTIC_COMBINABLE_CHANNELS = [
    ub.oset(['B04', 'B03', 'B02']),  # for onera
    ub.oset(['matseg_1', 'matseg_2', 'matseg_3']),  # hack
    # ub.oset(['snow_or_ice_field', 'built_up', 'grassland']),  # hack
]

CONFUSION_COLOR_SCHEME = {
    'TN': 'black',
    # 'TP': 'white',
    # 'TP': 'snow',  # off white
    'TP': 'whitesmoke',  # off white
    'FN': 'teal',
    'FP': 'red',
}


def dummy_legend():
    # hack to make a legend for slides

    label_to_color = {
        cat['name']: cat['color']
        for cat in CATEGORIES
    }
    label_to_color = ub.dict_subset(label_to_color, {
        'Post Construction', 'Site Preparation', 'Active Construction',
        'No Activity', 'Unknown'})

    import kwplot
    kwplot.autompl()
    img = kwplot.make_legend_img(CONFUSION_COLOR_SCHEME)
    kwplot.imshow(img)

    img = kwplot.make_legend_img(label_to_color)
    kwplot.imshow(img)


def build_image_header_text(**kwargs):
    """
    A heuristic for what sort of info is useful to plot on the header of an
    image.

    Kwargs:
        img
        coco_dset
        vidname,
        _header_extra

        gid,
        frame_index,
        dset_idstr,
        name,
        sensor_coarse,
        date_captured

    Example:
        >>> from watch.heuristics import *  # NOQA
        >>> img = {
        >>>     'id': 1,
        >>>     'frame_index': 0,
        >>>     'date_captured': '2020-01-01',
        >>>     'name': 'BLARG',
        >>>     'sensor_coarse': 'Sensor1',
        >>> }
        >>> kwargs = {
        >>>     'img': img,
        >>>     'dset_idstr': '',
        >>>     'name': '',
        >>>     '_header_extra': None,
        >>> }
        >>> header_lines = build_image_header_text(**kwargs)
        >>> print('header_lines = {}'.format(ub.repr2(header_lines, nl=1)))
    """
    img = kwargs.get('img', {})
    _header_extra = kwargs.get('_header_extra', None)
    dset_idstr = kwargs.get('dset_idstr', '')

    def _multi_get(key, default=ub.NoParam, *dicts):
        # try to lookup from multiple dictionaries
        found = default
        for d in dicts:
            if key in d:
                found = d[key]
                break
        if found is ub.NoParam:
            raise Exception
        return found

    sensor_coarse = _multi_get('sensor_coarse', 'unknown', kwargs, img)
    # name = _multi_get('name', 'unknown', kwargs, img)

    date_captured = _multi_get('date_captured', '', kwargs, img)
    frame_index = _multi_get('frame_index', None, kwargs, img)
    gid = _multi_get('id', None, kwargs, img)
    image_name = _multi_get('name', '', kwargs, img)

    vidname = None
    if 'vidname' in kwargs:
        vidname = kwargs['vidname']
    else:
        coco_dset = kwargs.get('coco_dset', None)
        if coco_dset is not None:
            vidname = coco_dset.index.videos[img['video_id']]['name']

    image_id_parts = []
    image_id_parts.append(f'gid={gid}')
    image_id_parts.append(f'frame_index={frame_index}')
    image_id_part = ', '.join(image_id_parts)

    header_line_infos = []
    header_line_infos.append([vidname, image_id_part, _header_extra])
    header_line_infos.append([dset_idstr])
    header_line_infos.append([image_name])
    header_line_infos.append([sensor_coarse, date_captured])
    header_lines = []
    for line_info in header_line_infos:
        header_line = ' '.join([p for p in line_info if p])
        header_line = header_line.replace('\\n', '\n')  # hack
        if header_line:
            header_lines.append(header_line)
    return header_lines


# NOTES ON QUALITY / CLOUDMASK
# https://github.com/GERSL/Fmask#46-version
# The cloudmask band is a class-idx based raster with labels
# 0 => clear land pixel
# 1 => clear water pixel
# 2 => cloud shadow
# 3 => snow
# 4 => cloud
# 255 => no observation

# However, in my data I seem to see:
# Unique values   8,  16,  65, 128

# These are specs
# https://smartgitlab.com/TE/standards/-/wikis/Data-Output-Specifications#quality-band
# TODO: this could be a specially handled frame like ASI.
# QA Information Moved to ~/code/watch/watch/tasks/fusion/datamodules/qa_bands.py
QUALITY_BITS = ub.udict({
    'TnE'           : 1 << 0,  # T&E binary mask
    'dilated_cloud' : 1 << 1,
    'cirrus'        : 1 << 2,
    'cloud'         : 1 << 3,
    'cloud_shadow'  : 1 << 4,
    'snow'          : 1 << 5,
    'clear'         : 1 << 6,
    'water'         : 1 << 7,
})


# The main dataset codes currently in use.
DATASET_CODES = [
    # 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC',
    # 'Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC',
    'Drop4-BAS',
    'Drop4-SC',
]


DVC_FIND_EXPT_KWARGS = {
    'tags': 'phase2_expt', 'envvar': 'DVC_EXPT_DPATH', 'hardware': 'auto'}
DVC_FIND_DATA_KWARGS = {
    'tags': 'phase2_data', 'envvar': 'DVC_DATA_DPATH', 'hardware': 'auto'}


def auto_expt_dvc():
    import watch
    return watch.find_smart_dvc_dpath(**DVC_FIND_EXPT_KWARGS)


def auto_data_dvc():
    import watch
    return watch.find_smart_dvc_dpath(**DVC_FIND_DATA_KWARGS)


# We should be able to figure out a way to robustly introspect these
fit_param_keys = [
    'sensorchan',
    # 'channels',
    'time_steps',
    'chip_dims', 'chip_overlap', 'arch_name', 'optimizer',
    'time_sampling', 'time_span', 'true_multimodal',
    'accumulate_grad_batches', 'modulate_class_weights', 'tokenizer',
    'use_grid_positives', 'use_cloudmask', 'upweight_centers',
    'temporal_dropout', 'stream_channels', 'saliency_loss',
    'class_loss', 'init', 'learning_rate', 'decoder',
]


pred_param_keys = [
    'pred_tta_fliprot',
    'pred_tta_time',
    'pred_chip_overlap',
]


trk_param_keys = [
    'trk_thresh',
    'trk_morph_kernel',
    'trk_agg_fn',
    'trk_thresh_hysteresis',
    'trk_moving_window_size',
]


act_param_keys = [
    'trk_use_viterbi',
    'trk_thresh',
]


DSET_CODE_TO_GSD = {
    # DEPRECATE
    'Aligned-Drop3-L1': 10.0,
    'Aligned-Drop3-TA1-2022-03-10': 10.0,
    'Cropped-Drop3-TA1-2022-03-10': 1.0,
}
