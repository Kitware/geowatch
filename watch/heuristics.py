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
    {'name': 'positive_annotated_static', 'color': 'black'},
    {'name': 'positive_annotated', 'color': 'black'},
    {'name': 'positive_partial', 'color': 'black'},
    {'name': 'positive_pending', 'color': 'black'},
    {'name': 'positive_unbounded', 'color': 'darkviolet'},
    {'name': 'ignore', 'color': 'lightsalmon'},
    {'name': 'seen', 'color': 'cyan'},
    {'name': 'train', 'color': 'cyan'},
    # Note: colors for these status labels are undefined, using neutral gray
    {'name': 'positive_excluded', 'color': 'gray'},
    {'name': 'negative', 'color': 'gray'},
    {'name': 'negative_unbounded', 'color': 'gray'},
]


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


# Mapping of annotation status to the kwcoco category name
# Used in project annotations
PHASE_STATUS_TO_KWCOCO_CATNAME = {
    'ignore': 'ignore',
    'negative': 'negative',
    'negative_unbounded': 'negative',
    'positive_excluded': 'ignore',  # This is positive, but is not "big" enough
    'positive_unbounded': 'positive',  # Start or end date might not be defined
    'positive_pending': 'positive',  # Does not have phase labels
    'positive_partial': 'positive',  # Does not have phase labels
    'positive_annotated': None,  # This must have a category already do not map
    'positive_annotated_static': None,  # This must have a category already do not map
}

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
        if header_line:
            header_lines.append(header_line)
    return header_lines
