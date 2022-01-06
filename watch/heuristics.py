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


CATEGORIES = [
    {'name': 'No Activity', 'color': 'tomato'},
    {'name': 'Site Preparation', 'color': 'gold'},
    {'name': 'Active Construction', 'color': 'lime'},
    {'name': 'Post Construction', 'color': 'darkturquoise'},
    {'name': 'Unknown', 'color': 'blueviolet'},
    {'name': 'ignore', 'color': 'slategray'},
    {'name': 'negative', 'color': 'orangered'},
    {'name': 'positive', 'color': 'olive'},
]


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
