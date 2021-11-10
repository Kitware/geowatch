# -*- coding: utf-8 -*-
"""
In the interest of development speed we can't always be perfect about correctly
passing the right metadata to functions that need it. This module serves as a
place to store hard-coded heuristics so we are explicit about where we are
cutting corners or using magic numbers. The idea is that  this will make it
easier for us to go back and make the code robust.
"""


# Might need to split this up into a finger-grained structure
IGNORE_CLASSNAMES = {
    'clouds', 'occluded',
    'ignore', 'unknown',

}

BACKGROUND_CLASSES = {
    'background', 'No Activity', 'Post Construction',
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
]
