"""
Describe how to interpret QA bands.

References:
    https://smart-research.slack.com/?redir=%2Ffiles%2FU028UQGN1N0%2FF04B998ANRL%2Faccenture_ta1_productdoc_phaseii_20211117.pptx%3Forigin_team%3DTN3QR7WAH%26origin_channel%3DC03QTAXU7GF
"""
import ubelt as ub
import functools
import operator
import numpy as np
import math
from watch.utils import util_pattern


def _dump_qa_debug_vid():
    """
    Make human interpretable sequences of QA bands and RGB data.
    """
    import kwcoco
    import watch
    import kwimage
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = data_dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
    dset = kwcoco.CocoDataset(coco_fpath)

    videos = dset.videos()
    images = videos.images[0]

    dump_dpath = ub.Path('_dump_qa_debug_vid').ensuredir()

    for idx, gid in enumerate(images):
        coco_img = dset.coco_image(gid)
        # Load some quality and rgb data
        qa_delayed = coco_img.delay('cloudmask', interpolation='nearest', antialias=False)
        rgb_delayed = coco_img.delay('red|green|blue')
        quality_im = qa_delayed.finalize()
        rgb_canvas = kwimage.normalize_intensity(rgb_delayed.finalize(nodata_method='float'))
        sensor = coco_img.img.get('sensor_coarse')
        print(f'sensor={sensor}')
        # Use the spec to draw it
        from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
        table = QA_SPECS.find_table('ACC-1', 'L8')
        #table = QA_SPECS.find_table('FMASK', '*')
        #table = QA_SPECS.find_table('Phase1_QA', '*')
        drawings = table.draw_labels(quality_im)
        qa_canvas = drawings['qa_canvas']
        legend = drawings['legend']
        canvas = kwimage.stack_images([rgb_canvas, qa_canvas, legend], axis=1)

        spec_name = table.spec['qa_spec_name']
        canvas = kwimage.draw_header_text(canvas, sensor + ' ' + spec_name)

        _kw = ub.compatible({'on_value': 0.3}, kwimage.fill_nans_with_checkers)
        canvas = kwimage.fill_nans_with_checkers(canvas, **_kw)

        fname = f'frame_{idx:08d}.jpg'
        fpath = dump_dpath / fname
        kwimage.imwrite(fpath, kwimage.ensure_uint255(canvas))

    #canvas = kwimage.stack_images([canvas, ], axis=0)
    import kwplot
    kwplot.autoplt()
    kwplot.imshow(canvas)


class QA_SpecMixin:

    def draw_labels(table, quality_im, legend='separate'):
        """

        The doctest can be used to debug cloudmasks for the datasets

        Ignore:
            >>> import kwcoco
            >>> import watch
            >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
            >>> coco_fpath = dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> gid = dset.images()[18]
            >>> coco_img = dset.coco_image(gid)
            >>> # Load some quality and rgb data
            >>> qa_delayed = coco_img.delay('cloudmask', interpolation='nearest', antialias=False)
            >>> rgb_delayed = coco_img.delay('red|green|blue')
            >>> quality_im = qa_delayed.finalize()
            >>> rgb_canvas = kwimage.normalize_intensity(rgb_delayed.finalize(nodata_method='float'))
            >>> sensor = coco_img.img.get('sensor_coarse')
            >>> print(f'sensor={sensor}')
            >>> # Use the spec to draw it
            >>> from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
            >>> table = QA_SPECS.find_table('ACC-1', 'L8')
            >>> #table = QA_SPECS.find_table('FMASK', '*')
            >>> #table = QA_SPECS.find_table('Phase1_QA', '*')
            >>> drawings = table.draw_labels(quality_im)
            >>> qa_canvas = drawings['qa_canvas']
            >>> legend = drawings['legend']
            >>> canvas = kwimage.stack_images([rgb_canvas, qa_canvas, legend], axis=1)
            >>> canvas = kwimage.draw_header_text(canvas, sensor)
            >>> #canvas = kwimage.stack_images([canvas, ], axis=0)
            >>> import kwplot
            >>> kwplot.autoplt()
            >>> kwplot.imshow(canvas)
        """
        import kwimage
        import kwarray
        import numpy as np
        import kwplot
        qavals_to_count = ub.dict_hist(quality_im.ravel())

        unique_qavals = list(qavals_to_count.keys())

        # For the QA band lets assign a color to each category
        colors = kwimage.Color.distinct(len(qavals_to_count))
        qval_to_color = dict(zip(unique_qavals, colors))

        qval_to_desc = table.describe_values(unique_qavals)
        quality_im = kwarray.atleast_nd(quality_im, 3)

        # Colorize the QA bands
        colorized = np.empty(quality_im.shape[0:2] + (3,), dtype=np.float32)
        for qabit, color in qval_to_color.items():
            mask = quality_im[:, :, 0] == qabit
            colorized[mask] = color

        # Because the QA band is categorical, we should be able to make a short
        qa_canvas = colorized

        label_to_color = ub.udict(qval_to_color).map_keys(qval_to_desc.__getitem__)
        legend = kwplot.make_legend_img(label_to_color)  # Make a legend

        drawings = {
            'qa_canvas': qa_canvas,
            'legend': legend,
        }
        return drawings


class QA_BitSpecTable(QA_SpecMixin):
    """
    Bit tables are more efficient because we can reduce over the query input

    Example:
        >>> from watch.tasks.fusion.datamodules import qa_bands
        >>> import kwimage
        >>> # Lookup a table for this spec
        >>> self = qa_bands.QA_SPECS.find_table('ACC-1', 'S2')
        >>> assert isinstance(self, qa_bands.QA_BitSpecTable)
        >>> # Make a quality image with every value
        >>> pure_patches = [np.zeros((32, 32), dtype=np.int16) + val for val in self.name_to_value.values()]
        >>> # Also add in a few mixed patches
        >>> mixed_patches = [
        >>>     pure_patches[0] | pure_patches[1],
        >>>     pure_patches[4] | pure_patches[1],
        >>>     pure_patches[3] | pure_patches[5],
        >>>     pure_patches[0] | pure_patches[4],
        >>>     pure_patches[3] | pure_patches[4],
        >>> ]
        >>> patches = pure_patches + mixed_patches
        >>> quality_im = kwimage.stack_images_grid(patches)
        >>> # The mask_any method makes a mask where any of the semantically given labels will be masked
        >>> query_names = ['cloud']
        >>> is_iffy = self.mask_any(quality_im, ['cloud', 'cirrus'])
        >>> drawings = self.draw_labels(quality_im)  # visualize
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> qa_canvas = drawings['qa_canvas']
        >>> legend = drawings['legend']
        >>> kwplot.imshow(is_iffy, pnum=(1, 3, 1), title=f'mask matching {query_names}')
        >>> kwplot.imshow(qa_canvas, pnum=(1, 3, 2), title='qa bits')
        >>> kwplot.imshow(legend, pnum=(1, 3, 3), title='qa bit legend')
        >>> kwplot.set_figtitle(f"QA Spec: name={self.spec['qa_spec_name']} sensor={self.spec['sensor']}")
    """
    def __init__(table, spec):
        table.spec = spec
        table.name_to_value = ub.udict({
            item['qa_name']: 1 << item['bit_number']
            for item in table.spec['bits']
            if item.get('qa_name', None) is not None
        })

    def mask_any(table, quality_im, qa_names):
        bit_values = list((table.name_to_value & qa_names).values())
        iffy_bits = functools.reduce(operator.or_, bit_values)
        is_iffy = (quality_im & iffy_bits) > 0
        return is_iffy

    def describe_values(table, unique_qavals):
        """
        Get a human readable description of each value for a legend
        """
        bit_to_spec = {}
        for item in table.spec['bits']:
            bit_to_spec[item['bit_number']] = item

        val_to_desc = {}

        for val in unique_qavals:
            # For each value determine what bits are on
            if val >= 0:
                bit_positions = unpack_bit_positions(val)

                descs = []
                for bit_number in bit_positions:
                    bit_spec = bit_to_spec.get(bit_number, '?')
                    descs.append(bit_spec['qa_description'])

                parts = {}
                parts['value'] = val
                parts['bits'] = '|'.join(list(map(str, bit_positions)))
                parts['desc'] = ',\n'.join(descs)
            else:
                parts = {}
                parts['value'] = val
                parts['bits'] = '---'
                parts['desc'] = 'nodata'
            val_to_desc[val] = ub.repr2(parts, compact=1, nobr=1, nl=True, si=1, sort=0)
        return val_to_desc


def unpack_bit_positions(val, itemsize=None):
    """
    Given an integer value, return the positions of the on bits.

    Args:
        val (int): a signed or unsigned integer

        itemsize (int | None):
            Number of bytes used to represent the integer. E.g. 1 for a uint8 4
            for an int32. If unspecified infer the smallest number of bytes
            needed, but warning this may produce ambiguous results for negative
            numbers.

    Returns:
        List[int]: the indexes of the 1 bits.

    Note:
        This turns out to be faster than a numpy or lookuptable strategy I
        tried.  See github.com:Erotemic/misc/learn/bit_conversion.py

    Example:
        >>> unpack_bit_positions(0)
        []
        >>> unpack_bit_positions(1)
        [0]
        >>> unpack_bit_positions(-1)
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>> unpack_bit_positions(-1, itemsize=2)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        >>> unpack_bit_positions(9)
        [0, 3]
        >>> unpack_bit_positions(2132)
        [2, 4, 6, 11]
        >>> unpack_bit_positions(-9999)
        [0, 4, 5, 6, 7, 11, 12, 14, 15]
        >>> unpack_bit_positions(np.int16(-9999))
        [0, 4, 5, 6, 7, 11, 12, 14, 15]
        >>> unpack_bit_positions(np.int16(-1))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    """
    is_negative = val < 0
    if is_negative:
        if itemsize is None:
            try:
                bit_length = val.bit_length() + 1
                itemsize = math.ceil(bit_length / 8.0)  # bytelength
            except AttributeError:
                # Probably a numpy type
                itemsize = val.dtype.itemsize
        neg_position = (itemsize * 8) - 1
        # special logic for negatives to get twos compliment repr
        max_val = 1 << neg_position
        val_ = max_val + val
    else:
        val_ = val
    binary_string = '{:b}'.format(val_)[::-1]
    bit_positions = [pos for pos, char in enumerate(binary_string)
                     if char == '1']
    if is_negative:
        bit_positions.append(neg_position)
    return bit_positions


class QA_ValueSpecTable(QA_SpecMixin):
    """
    Value tables are less efficient
    """
    def __init__(table, spec):
        table.spec = spec
        table.name_to_value = {
            item['qa_name']: item['value']
            for item in table.spec['values']
            if item.get('qa_name', None) is not None
        }

    def mask_any(table, quality_im, qa_names):
        iffy_values = list((table.name_to_value & qa_names).values())
        is_iffy = np.logical_or.reduce([quality_im == value for value in iffy_values])
        return is_iffy

    def describe_values(table, unique_qavals):
        """
        Get a human readable description of each value for a legend
        """

        val_to_spec = {}
        for item in table.spec['values']:
            val_to_spec[item['value']] = item
        val_to_desc = {}
        for val in unique_qavals:
            # For each value determine what bits are on
            spec = val_to_spec.get(val, None)
            parts = {}
            parts['value'] = val
            if spec is None:
                val_to_desc[val] = '?'
            else:
                parts['desc'] = spec['qa_description']
            val_to_desc[val] = ub.repr2(parts, compact=1, nobr=1, nl=True, si=1, sort=0)
        return val_to_desc


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

    def find_table(self, spec_name='*', sensor='*'):
        results = list(self.query_table(spec_name=spec_name, sensor=sensor))
        if len(results) != 1:
            raise AssertionError(f'{len(results)} - {spec_name}, {sensor}')
        table = results[0]
        return table

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
