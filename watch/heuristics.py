"""
In the interest of development speed we can't always be perfect about correctly
passing the right metadata to functions that need it. This module serves as a
place to store hard-coded heuristics so we are explicit about where we are
cutting corners or using magic numbers. The idea is that  this will make it
easier for us to go back and make the code robust.
"""
import ubelt as ub


# Might need to split this up into a finger-grained structure
IGNORE_CLASSNAMES = {
    'clouds', 'occluded',
    'ignore', 'unknown', 'Unknown',

}

BACKGROUND_CLASSES = {
    'background', 'No Activity', 'Post Construction', 'negative',
}


# These classes are used in BAS, but not in AC/SC
UNDISTINGUISHED_CLASSES = {
    'positive',
}


# # FIXME: Hard-coded category aliases.
# # The correct way to handle these would be to have some information in the
# # kwcoco category dictionary that specifies how the categories should be
# # interpreted.
# _HEURISTIC_CATEGORIES = {

#     'background': {'background', 'No Activity', 'Post Construction'},

#     'pre_background': {'No Activity'},
#     'post_background': {'Post Construction'},

#     'ignore': {'ignore', 'Unknown', 'clouds'},
# }


# https://smartgitlab.com/TE/annotations/-/wikis/Alternate-Site-Type
HUERISTIC_STATUS_DATA = [
    {'name': 'positive_annotated', 'color': 'olive'},
    {'name': 'positive_partial', 'color': 'limegreen'},
    {'name': 'positive_pending', 'color': 'seagreen'},
    {'name': 'positive_excluded', 'color': 'darkgreen'},
    {'name': 'positive_unbounded', 'color': 'steelblue'},
    {'name': 'negative', 'color': 'orangered'},
    {'name': 'negative_unbounded', 'color': 'deeppink'},
    {'name': 'ignore', 'color': 'purple'},
]

# mapping for "placeholder" categories "positive", "negative", "ignore"
# for statuses without an activity label.
# The "positive_annotated" status should always have an activity label.
STATUS_TO_CAT = {
    'ignore':
        'ignore',
    'positive_excluded':
        # This is positive, but is not "big" enough
        'ignore',
    'positive_unbounded':
        # Start or end date might not be defined.
        'positive',
    'positive_pending':
        # Does not have phase labels
        'positive',
    'positive_partial':
        # Might have phase labels
        'positive',
    # 'positive_annotated':
        # Has phase labels
        # assert catname is not None
    'negative':
        'negative',
    'negative_unbounded':
        'negative',
}

# metrics-and-test-framework/evaluation.py:1684
CATEGORIES_SCORED = [
    {'name': 'Site Preparation', 'color': 'gold'},
    {'name': 'Active Construction', 'color': 'lime'},
    {'name': 'Post Construction', 'color': 'darkturquoise'},
]

CATEGORIES_POSITIVE = CATEGORIES_SCORED + [
    {'name': 'positive', 'color': 'olive'},
]

CATEGORIES_NEGATIVE = [
    {'name': 'No Activity', 'color': 'tomato'},
    {'name': 'Unknown', 'color': 'blueviolet'},
    {'name': 'ignore', 'color': 'slategray'},
    {'name': 'negative', 'color': 'orangered'},
]

CATEGORIES = CATEGORIES_POSITIVE + CATEGORIES_NEGATIVE

CATEGORIES_DCT = {
        'positive': {
            'scored': CATEGORIES_SCORED,
            'unscored': CATEGORIES_POSITIVE[len(CATEGORIES_SCORED):],
        },
        'negative': {
            'scored': [],
            'unscored': CATEGORIES_NEGATIVE
        }
}

# 'name' field only
CNAMES_DCT = {
    k1: {k2: [cat['name'] for cat in cats]
         for k2, cats in dct.items()}
    for k1, dct in CATEGORIES_DCT.items()
}


def ensure_heuristic_colors(coco_dset):
    for hcat in CATEGORIES:
        cat = coco_dset.index.name_to_cat.get(hcat['name'], None)
        if cat is not None:
            if cat.get('color', None) is None:
                cat['color'] = hcat['color']


HUERISTIC_COMBINABLE_CHANNELS = [
    ub.oset(['B04', 'B03', 'B02']),  # for onera
    ub.oset(['matset_1', 'matset_2', 'matset_3']),  # hack
    ub.oset(['snow_or_ice_field', 'built_up', 'grassland']),  # hack
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
}
