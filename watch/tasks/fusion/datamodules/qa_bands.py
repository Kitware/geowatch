"""
Describe how to interpret QA bands.

References:
    https://smart-research.slack.com/?redir=%2Ffiles%2FU028UQGN1N0%2FF04B998ANRL%2Faccenture_ta1_productdoc_phaseii_20211117.pptx%3Forigin_team%3DTN3QR7WAH%26origin_channel%3DC03QTAXU7GF
"""
import ubelt as ub
import functools
import operator
import numpy as np
from watch.utils import util_pattern


class QA_BitSpecTable:
    """
    Bit tables are more efficient because we can reduce over the query input
    """
    def __init__(table, spec):
        table.spec = spec
        table.name_to_value = {
            item['qa_name']: 1 << item['bit_number']
            for item in table.spec['bits']
            if item.get('qa_name', None) is not None
        }

    def mask_any(table, quality_im, qa_names):
        bit_values = list(ub.take(table.name_to_value, qa_names))
        iffy_bits = functools.reduce(operator.or_, bit_values)
        is_iffy = (quality_im & iffy_bits) > 0
        return is_iffy


class QA_ValueSpecTable:
    """
    Value tables are less efficient
    """
    def __init__(table, spec):
        table.spec = spec
        table.name_to_value = {
            item['qa_name']: 1 << item['value']
            for item in table.spec['values']
            if item.get('qa_name', None) is not None
        }

    def mask_any(table, quality_im, qa_names):
        iffy_values = list(ub.take(table.name_to_value, qa_names))
        is_iffy = np.logical_or.reduce([quality_im == value for value in iffy_values])
        return is_iffy


class QA_SpecRegistry:
    def __init__(self):
        self.tables = []

    def query_table(self, spec_name='*', sensor='*'):
        """
        Ignore:
            from watch.tasks.fusion.datamodules.qa_bands import *  # NOQA
            self = QA_SPECS
            spec_name = 'ACC-1'
            sensor = 'L8'
            table, = list(self.query_table(spec_name, sensor))
            qa_names = ['cloud']
            table.mask_for_any(qa_names)
        """
        spec_pat = util_pattern.Pattern.coerce(spec_name)
        sensor_pat = util_pattern.Pattern.coerce(sensor)
        for table in self.tables:
            f1 = sensor_pat.match(table.spec['sensor'])
            f2 = spec_pat.match(table.spec['qa_spec_name'])
            if f1 and f2:
                yield table

    def __iadd__(self, table):
        self.tables.append(table)
        return self


QA_SPECS = QA_SpecRegistry()


QA_SPECS += QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'S2',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {'bit_number': 0, 'qa_name': 'combined', 'qa_description': 'combined qa mask', 'bit_value': [{'value': 1, 'description': 'use-pixel'}]},
        {'bit_number': 1, 'qa_name': 'cloud', 'qa_description': 'cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 2, 'qa_name': 'cloud_adjacent', 'qa_description': 'adjacent to cloud/shadow', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 3, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 4, 'qa_name': 'ice', 'qa_description': 'snow / ice', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 5, 'qa_name': 'water', 'qa_description': 'water', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 6, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {'bit_number': 7, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {'bit_number': 8, 'qa_name': 'filled', 'qa_description': 'filled value', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 9, 'qa_description': 'reserved for future use'},
        {'bit_number': 10, 'qa_description': 'reserved for future use'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'},
    ]
})


QA_SPECS += QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'L8',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {'bit_number': 0, 'qa_name': 'combined', 'qa_description': 'combined qa mask', 'bit_value': [{'value': 1, 'description': 'use-pixel'}]},
        {'bit_number': 1, 'qa_name': 'cloud', 'qa_description': 'cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 2, 'qa_name': 'cloud_adjacent', 'qa_description': 'adjacent to cloud/shadow', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 3, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 4, 'qa_name': 'ice', 'qa_description': 'snow / ice', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 5, 'qa_name': 'water', 'qa_description': 'water', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 6, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {'bit_number': 7, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {'bit_number': 8, 'qa_name': 'filled', 'qa_description': 'filled value', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 9, 'qa_description': 'reserved for future use'},
        {'bit_number': 10, 'qa_description': 'reserved for future use'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'},
    ]
})


"""
Note: In Ver 2 processing, WorldView VNIR products will include a 2nd QA file
containing QA information per multispectral band -Filename: *_QA2.tif -Format
same as original QA file
"""
QA_SPECS += QA_BitSpecTable({

    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'WV',
    'dtype': {'kind': 'u', 'itemsize': 2},

    'bits': [
        {'bit_number': 0, 'qa_name': 'combined', 'qa_description': 'combined qa mask', 'bit_value': [{'value': 1, 'description': 'use-pixel'}, {'value': 0, 'description': 'ignore-pixel'}]},
        {'bit_number': 1, 'qa_name': 'cloud', 'qa_description': 'cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 2, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 3, 'qa_name': 'thin_cloud', 'qa_description': 'thin cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 4, 'qa_description': 'reserved for future use', },
        {'bit_number': 7, 'qa_description': 'reserved for future use', },
        {'bit_number': 8, 'qa_description': 'filled value / suspicious pixel', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 9, 'qa_description': 'AOD Source', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 10, 'qa_description': 'climatology source', 'bit_value': [{'value': 1, 'description': 'climatology'}, {'value': 0, 'description': 'MODIS'}] },
        {'bit_number': 11, 'qa_description': 'reserved for future use', },
        {'bit_number': 12, 'qa_description': 'reserved for future use', },
        {'bit_number': 13, 'qa_description': 'reserved for future use', },
        {'bit_number': 14, 'qa_description': 'reserved for future use', },
        {'bit_number': 15, 'qa_description': 'reserved for future use', },
    ]
})


QA_SPECS += QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'PD',
    'dtype': {
        'kind': 'u',
        'itemsize': 2,  # in bytes
    },

    'bits': [
        {'bit_number': 0, 'qa_name': 'combined', 'qa_description': 'combined qa mask', 'bit_value': [ {'value': 1, 'description': 'use-pixel'}, {'value': 0, 'description': 'ignore-pixel'}, ], },
        {'bit_number': 1, 'qa_name': 'cloud', 'qa_description': 'cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 2, 'qa_description': 'reserved for future use', },
        {'bit_number': 7, 'qa_description': 'reserved for future use', },
        {'bit_number': 8, 'qa_name': 'filled', 'qa_description': 'filled value / suspicious pixel', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 9, 'qa_description': 'AOD Source', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}] },
        {'bit_number': 10, 'qa_description': 'climatology source', 'bit_value': [{'value': 1, 'description': 'climatology'}, {'value': 0, 'description': 'MODIS'}] },
        {'bit_number': 11, 'qa_description': 'reserved for future use', },
        {'bit_number': 12, 'qa_description': 'reserved for future use', },
        {'bit_number': 13, 'qa_description': 'reserved for future use', },
        {'bit_number': 14, 'qa_description': 'reserved for future use', },
        {'bit_number': 15, 'qa_description': 'reserved for future use', },
    ]
})


# https://github.com/GERSL/Fmask#46-version
QA_SPECS += QA_ValueSpecTable({
    'qa_spec_name': 'FMASK',
    'qa_spec_date': '???',
    'sensor': '*',
    'dtype': {'kind': 'u', 'itemsize': 1},
    # Note: this is different than a bit table.
    'values': [
        {'value': 0, 'qa_name': 'clear', 'qa_description': 'clear land pixel'},
        {'value': 1, 'qa_name': 'water', 'qa_description': 'clear water pixel'},
        {'value': 2, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow'},
        {'value': 3, 'qa_name': 'ice', 'qa_description': 'snow'},
        {'value': 4, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'value': 255, 'qa_description': 'no observation'},
    ]
})


# https://github.com/GERSL/Fmask#46-version
QA_SPECS += QA_BitSpecTable({
    'qa_spec_name': 'Phase1_QA',
    'qa_spec_date': '2022-03-28',
    'sensor': '*',
    'dtype': {'kind': 'u', 'itemsize': 1},
    # Note: this is different than a bit table.
    'bits': [
        { 'bit_number': 0, 'qa_name': 'combined', 'qa_description': 'TnE'},
        { 'bit_number': 1, 'qa_name': 'dilated_cloud', 'qa_description': 'dilated_cloud'},
        { 'bit_number': 2, 'qa_name': 'cirrus', 'qa_description': 'cirrus'},
        { 'bit_number': 3, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        { 'bit_number': 4, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud_shadow'},
        { 'bit_number': 5, 'qa_name': 'ice', 'qa_description': 'snow'},
        { 'bit_number': 6, 'qa_name': 'clear', 'qa_description': 'clear'},
        { 'bit_number': 7, 'qa_name': 'water', 'qa_description': 'water'},
    ]
})
