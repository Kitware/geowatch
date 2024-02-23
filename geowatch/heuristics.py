"""
In the interest of development speed we can't always be perfect about correctly
passing the right metadata to functions that need it. This module serves as a
place to store hard-coded heuristics so we are explicit about where we are
cutting corners or using magic numbers. The idea is that  this will make it
easier for us to go back and make the code robust.

References:
    https://gitlab.kitware.com/smart/annotations-wiki/-/blob/main/Annotation-Status-Types.md

"""
import ubelt as ub

# to make it easier to switch to "assets" when we finally deprecate auxiliary
# COCO_ASSETS_KEY = 'assets'
COCO_ASSETS_KEY = 'auxiliary'

HEURISTIC_START_STATES = {
    'No Activity',
}

HEURISTIC_END_STATES = {
    'Post Construction'
}


# # FIXME: Hard-coded category aliases.
# https://smartgitlab.com/TE/standards


"""
Status Data Info
================

The following is a long-form table, where list-items are rows and keys are
columns to describe how to interpret IARPA annotation status labels in
different cases. More information on status labels can be found in
[TEAnnotStatus]_ and in our internal fork [KWAnnotStatus]_.


Column Info:
    * status:
        the name of the site status label

    * color:
        what color to use for this status in visualizations. These are
        mostly taken from [EvalMetricColors]_, but in cases where they are
        undefined we make them up.

    * kwcoco_catname:
        what category to map this status to in the kwcoco dataset (and thus
        what will be learned), Note: a status is different than a phase label,
        and annotations with phase labels may overwrite the kwcoco category
        defined here. This is used in geowatch reproject.

    * positive_match_confusion
        This is the label the truth is given when it has some match in our set
        of positive predictions.  Denote what type of confusion a truth status
        incurs when it is matched.


The following code prints a concice version of this table and shows a legend
with colors.

.. code:: python

    from geowatch.heuristics import *  # NOQA
    import pandas as pd
    import rich
    df = pd.DataFrame(HUERISTIC_STATUS_DATA)
    rich.print(df.to_string())

    import kwplot
    kwplot.autompl()
    status_to_color = {r['status']: r['color'] for r in HUERISTIC_STATUS_DATA}
    img = kwplot.make_legend_img(status_to_color, dpi=300)
    kwplot.imshow(img, fnum=1)


References
----------
.. [TEAnnotStatus] https://smartgitlab.com/TE/annotations/-/wikis/Annotation-Status-Types
.. [KWAnnotStatus] https://gitlab.kitware.com/smart/annotations-wiki/-/blob/main/Annotation-Status-Types.md
.. [EvalMetricColors] https://smartgitlab.com/TE/metrics-and-test-framework/-/blob/main/iarpa_smart_metrics/evaluation.py#L1205

"""
HUERISTIC_STATUS_DATA = [
    ### Misc Rows
    {
        'status': 'seen',
        'color': 'cyan',
        'kwcoco_catname': None,
        'positive_match_confusion': 'gt_seen',
    },
    {
        'status': 'train',
        'color': 'cyan',
        'kwcoco_catname': None,
        'positive_match_confusion': 'gt_seen',
    },

    ### Ignore Rows
    {
        'status': 'ignore',
        'color': 'lightsalmon',
        'kwcoco_catname': 'ignore',
        'positive_match_confusion': 'gt_ignore',
    },

    ### Negative Rows
    # Note: 'colors for these status labels are undefined, using neutral gray
    {
        'status': 'negative',
        'color': 'gray',
        'kwcoco_catname': 'negative',
        'positive_match_confusion': 'gt_false_pos',
    },
    {
        'status': 'negative_unbounded',
        'color': 'gray',
        'kwcoco_catname': 'negative',
        'positive_match_confusion': 'gt_false_pos',
    },

    ### Positive Rows
    {
        'status': 'positive_annotated',
        'color': 'black',
        # Requires a category phase label, do not use status to map
        'kwcoco_catname': None,
        'positive_match_confusion': 'gt_true_pos',
    },
    {
        'status': 'positive_annotated_static',
        'color': 'black',
        # Requires a category phase label, do not use status to map
        'kwcoco_catname': None,
        'positive_match_confusion': 'gt_true_pos',
    },
    {
        'status': 'positive_excluded',
        'color': 'gray',
        # This is positive, but is not "big" enough
        # Setting this to "positive" by default, because it is ignored at
        # evaluation time.
        # 'kwcoco_catname': 'ignore',
        'kwcoco_catname': 'positive',
        'positive_match_confusion': 'gt_false_pos',
    },
    {
        'status': 'positive_partial',
        'color': 'black',
        'kwcoco_catname': 'positive',
        'positive_match_confusion': 'gt_true_pos',
    },
    {
        'status': 'positive_pending',
        'color': 'black',
        'kwcoco_catname': 'positive',
        'positive_match_confusion': 'gt_true_pos',
    },

    {
        'status': 'positive_unbounded',
        'color': 'darkviolet',
        'kwcoco_catname': 'positive',   # Start or end date might not be defined
        'positive_match_confusion': 'gt_positive_unbounded',
    },

    ### Transient Rows
    {
        'status': 'transient_positive',
        'color': 'steelblue',
        'kwcoco_catname': 'transient',
    },
    {
        'status': 'transient_negative',
        'color': 'rust',
        'kwcoco_catname': 'negative',
    },
    {
        'status': 'transient_excluded',
        'kwcoco_catname': 'ignore',
        'color': 'lightsalmon',
    },
    {
        'status': 'transient_ignore',
        'kwcoco_catname': 'ignore',
        'color': 'lightsalmon',
    },
    {
        'status': 'transient_pending',
        'kwcoco_catname': 'ignore',
        'color': 'lightsalmon',
    },

    ### Prediction Rows

    # TODO? Add alias of pending for "positive_pending"? For QFabric?
    {
        'status': 'system_confirmed',
        'color': 'kitware_blue'
    },

    {
        'status': 'system_rejected',
        'color': 'kitware_yellow'
    },
]

# Backwards compat
for row in HUERISTIC_STATUS_DATA:
    row['name'] = row['status']


# Convinience mappings used in reproject annotations
PHASE_STATUS_TO_KWCOCO_CATNAME = {
    row['name']: row['kwcoco_catname'] for row in HUERISTIC_STATUS_DATA
    if 'kwcoco_catname' in row
}

IARPA_STATUS_TO_INFO = {row['status']: row for row in HUERISTIC_STATUS_DATA}

# update HUERISTIC_STATUS_DATA
# for status, row in IARPA_STATUS_TO_INFO.items():
#     if status in PHASE_STATUS_TO_KWCOCO_CATNAME:
#         row['kwcoco_catname'] = PHASE_STATUS_TO_KWCOCO_CATNAME[status]

# for status, row in IARPA_STATUS_TO_INFO.items():
#     if status in PHASE_STATUS_TO_MATCHED_CONFUSION:
#         row['positive_match_confusion'] = PHASE_STATUS_TO_MATCHED_CONFUSION[status]


if 0:
    import pandas as pd
    print(pd.DataFrame(HUERISTIC_STATUS_DATA).to_string())


IARPA_REAL_STATUS = {
    'positive': ["positive_annotated", "positive_annotated_static", "positive_partial", "positive_pending"],
    'negative': ["positive_excluded", "negative", "negative_unbounded"],
    'ignore': 'ignore',
}


"""
For official color definitions see:

    :func:`iarpa_smart_metrics.evaluation.Evaluation.get_sm_color` and
    :func:`iarpa_smart_metrics.evaluation.Evaluation.get_gt_color`
"""
IARPA_CONFUSION_COLORS = {}
IARPA_CONFUSION_COLORS['gt_true_neg'] = 'green'  # no IARPA color for this, make one up.
IARPA_CONFUSION_COLORS['gt_true_pos'] = 'lime'
IARPA_CONFUSION_COLORS['gt_false_pos'] = 'red'
IARPA_CONFUSION_COLORS['gt_false_neg'] = 'black'
IARPA_CONFUSION_COLORS['gt_positive_unbounded'] = "darkviolet"
IARPA_CONFUSION_COLORS['gt_ignore'] = "lightsalmon"
IARPA_CONFUSION_COLORS['gt_seen'] = "gray"
IARPA_CONFUSION_COLORS['sm_pos_match'] = "aquamarine"
IARPA_CONFUSION_COLORS['sm_partially_wrong'] = "orange"
IARPA_CONFUSION_COLORS['sm_completely_wrong'] = "magenta"

IARPA_CONFUSION_COLORS['sm_ignore'] = "lightsalmon"  # no IARPA color for this, make one up


def iarpa_assign_truth_confusion(truth_status, has_positive_match):
    """
    Example:
        >>> from geowatch.heuristics import *  # NOQA
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
        elif truth_status in {'transient_positive'}:
            gt_cfsn = 'gt_false_neg'

    return gt_cfsn


def iarpa_assign_pred_confusion(truth_match_statuses):
    """
    Example:
        >>> from geowatch.heuristics import *  # NOQA
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
    if truth_cfsns & {'gt_true_pos', 'gt_positive_unbounded'}:
        if 'gt_false_pos' in truth_cfsns:
            pred_cfsn = 'sm_partially_wrong'
        else:
            pred_cfsn = 'sm_pos_match'
    elif 'gt_false_pos' in truth_cfsns:
        pred_cfsn = 'sm_completely_wrong'
    else:
        if set(truth_cfsns) == {'gt_ignore'}:
            pred_cfsn = 'sm_ignore'
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
    {
        'name': 'background',
        'color': 'black',
        'scored': False,
        'required': True,
    },

    {
        'name': 'ignore',
        'color': 'violet',
        'scored': False,
        'required': True,
    },

    {
        'name': 'Unknown',
        'color': 'blueviolet',
        'scored': False,
    },

    {
        'name': 'positive',
        'color': 'palegreen',
        'scored': False,
    },

    {
        'name': 'negative',
        'color': 'gray',
        'scored': False,
    },

    {
        'name': 'Site Preparation',
        'color': 'gold',
        'scored': True,
    },
    {
        'name': 'Active Construction',
        'color': 'lime',
        'scored': True,
    },
    # Conditional classes
    {
        'name': 'Post Construction',
        'color': 'darkturquoise',
        'scored': True,
    },
    {
        'name': 'No Activity',
        'color': 'tomato',
        'scored': True,
    },
    # Transient
    {
        'name': 'transient',
        'color': 'steelblue',
        'scored': True,
    },
    # Transient
    {
        'name': 'transient',
        'color': 'steelblue',
        'scored': True,
    },
]


def hack_track_categories(track_catnames, task):
    """
    Returns:
        List[str]: Modified categories

    Example:
        >>> from geowatch.heuristics import *  # NOQA
        >>> basis = {
        >>>     #'task': ['class', 'saliency'],
        >>>     'task': ['class'],
        >>>     'track_catnames': [
        >>>         ['No Activity'],
        >>>         ['Post Construction'],
        >>>         ['Post Construction', 'Post Construction', ],
        >>>         ['Post Construction', 'Post Construction', 'Post Construction', ],
        >>>         ['No Activity', 'ignore', 'ignore'],
        >>>         ['No Activity', 'Post Construction'],
        >>>         ['No Activity', 'Site Preparation', 'Post Construction'],
        >>>     ],
        >>> }
        >>> for kw in ub.named_product(basis):
        >>>     task = kw['task']
        >>>     track_catnames = kw['track_catnames']
        >>>     kw['new_catnames'] = hack_track_categories(track_catnames, task)
        >>>     print('kw = {}'.format(ub.urepr(kw, nl=1)))

    Example:
        >>> from geowatch.heuristics import *  # NOQA
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

    main_classes = {'No Activity', 'Site Preparation', 'Active Construction', 'Post Construction'}

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


# Might need to split this up into a finer-grained structure
IGNORE_CLASSNAMES = {'ignore', 'Unknown'}
BACKGROUND_CLASSES = {'background'}
NEGATIVE_CLASSES = {'negative'}
UNDISTINGUISHED_CLASSES =  {'positive'}
CONTEXT_CLASSES = {'No Activity', 'Post Construction'}


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
        >>> from geowatch.heuristics import *  # NOQA
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
        - [ ] Consolidate with ~/code/watch/geowatch/tasks/fusion/utils :: category_tree_ensure_color
        - [ ] Consolidate with ~/code/watch/geowatch/utils/kwcoco_extensions :: category_category_colors
        - [ ] Consolidate with ~/code/watch/geowatch/heuristics.py :: ensure_heuristic_category_tree_colors
        - [ ] Consolidate with ~/code/watch/geowatch/heuristics.py :: ensure_heuristic_coco_colors

    Example:
        >>> from geowatch import heuristics
        >>> import kwcoco
        >>> classes = kwcoco.CategoryTree.coerce(['ignore', 'positive', 'Active Construction', 'foobar', 'Unknown', 'baz'])
        >>> heuristics.ensure_heuristic_category_tree_colors(classes)
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
    """
    from geowatch.heuristics import *  # NOQA
    dummy_legend()

    """
    import kwplot
    kwplot.autompl()
    img = kwplot.make_legend_img(CONFUSION_COLOR_SCHEME)
    kwplot.imshow(img, fnum=1)

    label_to_color = {
        cat['name']: cat['color']
        for cat in CATEGORIES
    }
    img = kwplot.make_legend_img(label_to_color)
    kwplot.imshow(img, fnum=2)

    label_to_color = ub.dict_subset(label_to_color, {
        'Post Construction', 'Site Preparation', 'Active Construction',
        'No Activity', 'Unknown'})
    img = kwplot.make_legend_img(label_to_color)
    kwplot.imshow(img, fnum=3)


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
        >>> from geowatch.heuristics import *  # NOQA
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
        >>> print('header_lines = {}'.format(ub.urepr(header_lines, nl=1)))
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
            video_id = img.get('video_id', None)
            if video_id is not None:
                vidname = coco_dset.index.videos[video_id]['name']
            else:
                vidname = 'loose-images'

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


# TODO: this could be a specially handled frame like ASI.
# NOTE:
# QA Information Moved to ~/code/watch/geowatch/tasks/fusion/datamodules/qa_bands.py

DVC_FIND_EXPT_KWARGS = {
    'tags': 'phase2_expt', 'envvar': 'DVC_EXPT_DPATH', 'hardware': 'auto'}
DVC_FIND_DATA_KWARGS = {
    'tags': 'phase2_data', 'envvar': 'DVC_DATA_DPATH', 'hardware': 'auto'}


def auto_expt_dvc():
    import geowatch
    return geowatch.find_dvc_dpath(**DVC_FIND_EXPT_KWARGS)


def auto_data_dvc():
    import geowatch
    return geowatch.find_dvc_dpath(**DVC_FIND_DATA_KWARGS)


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


PHASES = ['No Activity', 'Site Preparation', 'Active Construction', 'Post Construction']


# Represents default relative weights to give to each sensor in temporal
# sampling.
SENSOR_TEMPORAL_SAMPLING_VALUES = {
    'WV': 10,
    'WV1': 9,
    'S2': 1,
    'PD': 7,
    'L8': 0.3,
    'sensor1': 11,
    'sensor2': 7,
    'sensor3': 5,
    'sensor4': 3,
}


# Hard coded values for which regions are cleared
REGION_STATUS = [
    {'region_id': 'AE_R001', 'cleared': True},
    {'region_id': 'BH_R001', 'cleared': True},
    {'region_id': 'BR_R001', 'cleared': True},
    {'region_id': 'BR_R002', 'cleared': True},
    {'region_id': 'BR_R004', 'cleared': True},
    {'region_id': 'BR_R005', 'cleared': True},
    {'region_id': 'CH_R001', 'cleared': True},
    {'region_id': 'KR_R001', 'cleared': True},
    {'region_id': 'KR_R002', 'cleared': True},
    {'region_id': 'LT_R001', 'cleared': True},
    {'region_id': 'NZ_R001', 'cleared': True},
    {'region_id': 'PE_R001', 'cleared': True},
    {'region_id': 'US_R001', 'cleared': True},
    {'region_id': 'US_R004', 'cleared': True},
    {'region_id': 'US_R005', 'cleared': True},
]


# Mapping from our sensor names to the official T&E sensor names
SENSOR_TABLE = [
    {'te_name': 'WorldView', 'kit_name': 'WV'},
    {'te_name': 'Sentinel-2', 'kit_name': 'S2'},
    {'te_name': 'Landsat 7', 'kit_name': 'LE'},
    {'te_name': 'Landsat 8', 'kit_name': 'LC'},
    {'te_name': 'Landsat 8', 'kit_name': 'L8'},
    {'te_name': 'WorldView', 'kit_name': 'WV1'},
    {'te_name': 'Planet', 'kit_name': 'PD'},
]
{r['kit_name']: r['te_name'] for r in SENSOR_TABLE}
# {r['te_name']: r['kit_name'] for r in SENSOR_TABLE[::-1]}

TE_SENSOR_NAMES = {
    'WV': 'WorldView',
    'S2': 'Sentinel-2',
    'LE': 'Landsat 7',
    'LC': 'Landsat 8',
    'L8': 'Landsat 8',
    'WV1': 'WorldView',
    'PD': 'Planet',
}

SENSOR_TRACK_PRIORITY = {
    'WorldView': 6,
    'WorldView 1': 5,
    'Planet': 4,
    'Sentinel-2': 3,
    'Landsat 8': 2,
    'Landsat 7': 1
}


def normalize_sensors(coco_dset, sensor_warnings=True, format='te'):
    """
    Convert to / from internal representations or IAPRA sensor standards
    """
    from geowatch.heuristics import TE_SENSOR_NAMES
    sensor_dict = TE_SENSOR_NAMES
    good_sensors = set(sensor_dict.values())

    for img in coco_dset.dataset['images']:
        try:
            sensor = img['sensor_coarse']
            if sensor not in good_sensors:
                img['sensor_coarse'] = sensor_dict[sensor]
        except KeyError:
            if sensor_warnings:
                # name = img.get('name', img['file_name'])
                sensor = img.get('sensor_coarse', None)
                import warnings
                warnings.warn(
                    f'image has unknown sensor {sensor} in tag={coco_dset.tag}')

    return coco_dset


def extract_region_id(fname):
    """
    Example:
        >>> fname = 'foobar_KR_R001_otherstuff'
        >>> extract_region_id(fname)
    """
    import kwutil
    # Find a region pattern
    pat = kwutil.util_pattern.Pattern.coerce(r'([A-Z]+_[A-Z]\d+)', 'regex')
    found = pat.search(fname)
    name = found.groups()[0]
    return name


def register_known_fsspec_s3_buckets():
    """
    A workaround to handle requester pays information for particular s3
    endpoints. Ideally the user would be able to specify this mapping via the
    CLI, but for now lets just hack it in.

    We are not specifying the profile here, assuming that instead the user
    will use the ``AWS_DEFAULT_PROFILE`` environ.

    Note: the ``AWS_REQUEST_PAYER`` environ is only repsected by gdal, and this
    function does not impact gdal at all, so this environ needs to be set as
    well as calling this workaround.


    Ignore:
        from geowatch import heuristics
        heuristics.register_known_fsspec_s3_buckets()

        from geowatch.utils.util_fsspec import S3Path
        self = S3Path.coerce('/vsis3/usgs-landsat-ard/collection02')
        self.ls()
    """
    from geowatch.utils import util_fsspec
    util_fsspec.S3Path.register_bucket('s3://usgs-landsat-ard', requester_pays=True)
    util_fsspec.S3Path.register_bucket('s3://usgs-landsat', requester_pays=True)
