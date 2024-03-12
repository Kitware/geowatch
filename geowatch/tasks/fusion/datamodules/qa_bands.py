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
from kwutil import util_pattern


def _dump_qa_debug_vid():
    """
    Make human interpretable sequences of QA bands and RGB data.
    """
    import kwcoco
    import geowatch
    import kwimage
    data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = data_dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
    dset = kwcoco.CocoDataset(coco_fpath)

    videos = dset.videos()
    images = videos.images[0]

    dump_dpath = ub.Path('_dump_qa_debug_vid').ensuredir()

    for idx, gid in enumerate(images):
        coco_img = dset.coco_image(gid)
        # Load some quality and rgb data
        qa_delayed = coco_img.imdelay('cloudmask', interpolation='nearest', antialias=False)
        rgb_delayed = coco_img.imdelay('red|green|blue')
        quality_im = qa_delayed.finalize()
        rgb_canvas = kwimage.normalize_intensity(rgb_delayed.finalize(nodata_method='float'))
        sensor = coco_img.img.get('sensor_coarse')
        print(f'sensor={sensor}')
        # Use the spec to draw it
        from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS
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

    def draw_labels(table, quality_im, legend='separate', legend_dpi=96, verbose=0):
        """

        The doctest can be used to debug cloudmasks for the datasets

        Ignore:
            >>> import kwcoco
            >>> import geowatch
            >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data')
            >>> coco_fpath = dvc_dpath / 'Drop6/data_vali_split1.kwcoco.zip'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> gid = dset.images()[18]
            >>> coco_img = dset.coco_image(gid)
            >>> # Load some quality and rgb data
            >>> qa_delayed = coco_img.imdelay('cloudmask', interpolation='nearest', antialias=False)
            >>> rgb_delayed = coco_img.imdelay('red|green|blue')
            >>> quality_im = qa_delayed.finalize()
            >>> rgb_canvas = kwimage.normalize_intensity(rgb_delayed.finalize(nodata_method='float'))
            >>> sensor = coco_img.img.get('sensor_coarse')
            >>> print(f'sensor={sensor}')
            >>> # Use the spec to draw it
            >>> from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS
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

        Ignore:
            >>> import kwcoco
            >>> import geowatch
            >>> import kwimage
            >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
            >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data')
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> wv_gids = [g for g, s in dset.images().lookup('sensor_coarse', keepid=True).items() if s == 'WV']
            >>> gid = dset.images(wv_gids)[-3]
            >>> coco_img = dset.coco_image(gid)
            >>> # Load some quality and rgb data
            >>> qa_delayed = coco_img.imdelay('quality', interpolation='nearest', antialias=False)
            >>> rgb_delayed = coco_img.imdelay('red|green|blue')
            >>> quality_im = qa_delayed.finalize(interpolation='nearest', antialias=False)
            >>> rgb_raw = rgb_delayed.finalize(nodata_method='float')
            >>> rgb_canvas = kwimage.normalize_intensity(rgb_raw)
            >>> sensor = coco_img.img.get('sensor_coarse')
            >>> print(f'sensor={sensor}')
            >>> # Use the spec to draw it
            >>> from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS
            >>> table = QA_SPECS.find_table('ACC-1', 'WV')
            >>> #table = QA_SPECS.find_table('FMASK', '*')
            >>> #table = QA_SPECS.find_table('Phase1_QA', '*')
            >>> drawings = table.draw_labels(quality_im)
            >>> qa_canvas = drawings['qa_canvas']
            >>> legend = drawings['legend']
            >>> canvas = kwimage.stack_images([rgb_canvas, qa_canvas], axis=1)
            >>> canvas = kwimage.draw_header_text(canvas, sensor)
            >>> #canvas = kwimage.stack_images([canvas, ], axis=0)
            >>> import kwplot
            >>> kwplot.autoplt()
            >>> kwplot.imshow(canvas, fnum=1)
            >>> kwplot.imshow(legend, fnum=2)
        """
        import kwimage
        import kwarray
        import numpy as np
        import kwplot

        if verbose:
            print(f'Build quality image for {quality_im.shape}')

        _raw = quality_im.ravel()

        if quality_im.dtype.kind not in {'u', 'i'}:
            if verbose:
                print('Check for nan')
            is_nan = np.isnan(_raw)
            num_nan = is_nan.sum()
            _raw2 = _raw[~is_nan]
        else:
            num_nan = 0
            _raw2 = _raw

        if verbose:
            print('Counting unique values')

        # qavals_to_count = ub.dict_hist(_raw2)
        qavals, counts = np.unique(_raw2, return_counts=True)
        qavals_to_count = ub.dzip(qavals, counts)

        if num_nan:
            print('warning nan QA')
            # qavals_to_count[np.nan] = num_nan

        unique_qavals = list(qavals_to_count.keys())

        if verbose:
            print(f'Found {len(unique_qavals)} unique labels')

        max_labels = 32
        if len(qavals_to_count) > max_labels:
            print('WARNING: QA band has a lot of unique values')
            top_qvals = dict(list(ub.udict(qavals_to_count).sorted_values().items())[-max_labels:])
            unique_qavals = list(top_qvals)
            # qval_to_color = ub.udict(qval_to_color)
            # qval_to_color = qval_to_color.subdict(top_qvals)

        # For the QA band lets assign a color to each category
        colors = kwimage.Color.distinct(len(unique_qavals))
        qval_to_color = dict(zip(unique_qavals, colors))

        qval_to_desc = table.describe_values(unique_qavals)
        quality_im = kwarray.atleast_nd(quality_im, 3)

        # Colorize the QA bands
        if verbose:
            print('Colorizing')

        colorized = np.empty(quality_im.shape[0:2] + (3,), dtype=np.uint8)
        if len(qval_to_color) > 10:
            qa_iter = ub.ProgIter(qval_to_color.items(), total=len(qval_to_color), desc='complex QA')
        else:
            qa_iter = qval_to_color.items()
        for qabit, color in ub.ProgIter(qa_iter, desc='colorize', enabled=verbose):
            color255 = kwimage.Color.coerce(color).as255()
            mask = quality_im[:, :, 0] == qabit
            colorized[mask] = color255

        # Because the QA band is categorical, we should be able to make a short
        qa_canvas = colorized

        label_to_color = ub.udict(qval_to_color).map_keys(qval_to_desc.__getitem__)

        if verbose:
            print('Build legend')
        legend = kwplot.make_legend_img(label_to_color, dpi=legend_dpi)  # Make a legend

        if verbose:
            print('finished qa drawing')

        drawings = {
            'qa_canvas': qa_canvas,
            'legend': legend,
        }
        return drawings


class QA_BitSpecTable(QA_SpecMixin):
    """
    Bit tables are more efficient because we can reduce over the query input

    Example:
        >>> from geowatch.tasks.fusion.datamodules import qa_bands
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

    Example:
        >>> from geowatch.tasks.fusion.datamodules import qa_bands
        >>> import kwimage
        >>> # Lookup a table for this spec
        >>> self = qa_bands.QA_SPECS.find_table('qa_pixel', 'L8')
        >>> assert isinstance(self, qa_bands.QA_BitSpecTable)
        >>> # Make a quality image with every value
        >>> pure_patches = [np.zeros((32, 32), dtype=np.int16) + val for val in self.name_to_value.values()]
        >>> # Also add in a few mixed patches
        >>> mixed_patches = [
        >>>     pure_patches[0] | pure_patches[1],
        >>>     pure_patches[2] | pure_patches[1],
        >>> ]
        >>> patches = pure_patches + mixed_patches
        >>> quality_im = kwimage.stack_images_grid(patches)
        >>> # The mask_any method makes a mask where any of the semantically given labels will be masked
        >>> query_names = ['cloud']
        >>> is_iffy = self.mask_any(quality_im, ['cloud'])
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

        ranged_bits = []
        for item in table.spec.get('bit_ranges', []):
            if item.get('qa_name', None) is not None:
                ranged_bits.extend(item['bit_range'])
                # print('item = {}'.format(ub.urepr(item, nl=1)))
        table.ranged_bits = ranged_bits

    def mask_any(table, quality_im, qa_names):
        if quality_im.dtype.kind == 'f':
            raise ValueError('The quality mask should be an bitwise integer type, not a float')
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
                    if bit_number in table.ranged_bits:
                        continue
                    bit_spec = bit_to_spec.get(bit_number, '?')
                    if bit_spec == '?':
                        descs.append('?')
                    else:
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
            val_to_desc[val] = '───\n' + ub.urepr(parts, compact=1, nobr=1, nl=True, si=1, sort=0) + '\n───'
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
        table.name_to_value = ub.udict({
            item['qa_name']: item['value']
            for item in table.spec['values']
            if item.get('qa_name', None) is not None
        })

    def mask_any(table, quality_im, qa_names):
        if quality_im.dtype.kind == 'f':
            raise ValueError('The quality mask should be an integer type, not a float')
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
            val_to_desc[val] = ub.urepr(parts, compact=1, nobr=1, nl=True, si=1, sort=0)
        return val_to_desc


class QA_SpecRegistry(list):

    def query_table(self, spec_name='*', sensor='*'):
        """
        Ignore:
            from geowatch.tasks.fusion.datamodules.qa_bands import *  # NOQA
            self = QA_SPECS
            spec_name = 'ACC-1'
            sensor = 'L8'
            table, = list(self.query_table(spec_name, sensor))
            qa_names = ['cloud']
            table.mask_for_any(qa_names)
        """
        spec_pat = util_pattern.Pattern.coerce(spec_name)
        sensor_pat = util_pattern.Pattern.coerce(sensor)
        for table in self:
            matches_main_sensor = sensor_pat.match(table.spec['sensor'])
            maches_sensor_alias = any(
                sensor_pat.match(a) for a in table.spec.get('sensor_alias', []))
            matches_spec_name = spec_pat.match(table.spec['qa_spec_name'])
            matches_sensor = matches_main_sensor or maches_sensor_alias
            if matches_spec_name and matches_sensor:
                yield table

    def find_table(self, spec_name='*', sensor='*'):
        results = list(self.query_table(spec_name=spec_name, sensor=sensor))
        if len(results) != 1:
            raise AssertionError(f'{len(results)} - {spec_name}, {sensor}')
        table = results[0]
        return table


QA_SPECS = QA_SpecRegistry()


QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'S2',
    'sensor_alias': ['Sentinel-2'],
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {
            'bit_number': 0,
            'qa_name': 'combined',
            'qa_description': 'combined qa mask',
            'bit_value': [
                {'value': 1, 'description': 'use-pixel'}
            ]
        },
        {
            'bit_number': 1,
            'qa_name': 'cloud',
            'qa_description': 'cloud',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 2,
            'qa_name': 'cloud_adjacent',
            'qa_description': 'adjacent to cloud/shadow',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 3,
            'qa_name': 'cloud_shadow',
            'qa_description': 'cloud shadow',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 4,
            'qa_name': 'ice',
            'qa_description': 'snow / ice',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 5,
            'qa_name': 'water',
            'qa_description': 'water',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {'bit_number': 6, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {'bit_number': 7, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {
            'bit_number': 8,
            'qa_name': 'filled',
            'qa_description': 'filled value',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {'bit_number': 9, 'qa_description': 'reserved for future use'},
        {'bit_number': 10, 'qa_description': 'reserved for future use'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'L8',
    'sensor_alias': ['Landsat 8'],
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {
            'bit_number': 0,
            'qa_name': 'combined',
            'qa_description': 'combined qa mask',
            'bit_value': [
                {'value': 1, 'description': 'use-pixel'}
            ]
        },
        {
            'bit_number': 1,
            'qa_name': 'cloud',
            'qa_description': 'cloud',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 2,
            'qa_name': 'cloud_adjacent',
            'qa_description': 'adjacent to cloud/shadow',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 3,
            'qa_name': 'cloud_shadow',
            'qa_description': 'cloud shadow',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 4,
            'qa_name': 'ice',
            'qa_description': 'snow / ice',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 5,
            'qa_name': 'water',
            'qa_description': 'water',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {'bit_number': 6, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {'bit_number': 7, 'qa_name': None, 'qa_description': 'reserved for future use'},
        {
            'bit_number': 8,
            'qa_name': 'filled',
            'qa_description': 'filled value',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {'bit_number': 9, 'qa_description': 'reserved for future use'},
        {'bit_number': 10, 'qa_description': 'reserved for future use'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


"""
Note: In Ver 2 processing, WorldView VNIR products will include a 2nd QA file
containing QA information per multispectral band -Filename: *_QA2.tif -Format
same as original QA file
"""
QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'WV',
    'sensor_alias': ['WV1'],
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {
            'bit_number': 0,
            'qa_name': 'combined',
            'qa_description': 'combined qa mask',
            'bit_value': [
                {'value': 1, 'description': 'use-pixel'},
                {'value': 0, 'description': 'ignore-pixel'}
            ]
        },
        {
            'bit_number': 1,
            'qa_name': 'cloud',
            'qa_description': 'cloud',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 2,
            'qa_name': 'cloud_shadow',
            'qa_description': 'cloud shadow',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 3,
            'qa_name': 'thin_cloud',
            'qa_description': 'thin cloud',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {'bit_number': 4, 'qa_description': 'reserved for future use'},
        {'bit_number': 7, 'qa_description': 'reserved for future use'},
        {
            'bit_number': 8,
            'qa_description': 'filled value / suspicious pixel',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 9,
            'qa_description': 'AOD Source',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 10,
            'qa_description': 'climatology source',
            'bit_value': [
                {'value': 1, 'description': 'climatology'},
                {'value': 0, 'description': 'MODIS'}
            ]
        },
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ACC-1',
    'qa_spec_date': '2022-11-28',
    'sensor': 'PD',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {
            'bit_number': 0,
            'qa_name': 'combined',
            'qa_description': 'combined qa mask',
            'bit_value': [
                {'value': 1, 'description': 'use-pixel'},
                {'value': 0, 'description': 'ignore-pixel'}
            ]
        },
        {
            'bit_number': 1,
            'qa_name': 'cloud',
            'qa_description': 'cloud',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {'bit_number': 2, 'qa_description': 'reserved for future use'},
        {'bit_number': 7, 'qa_description': 'reserved for future use'},
        {
            'bit_number': 8,
            'qa_name': 'filled',
            'qa_description': 'filled value / suspicious pixel',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 9,
            'qa_description': 'AOD Source',
            'bit_value': [
                {'value': 1, 'description': 'yes'},
                {'value': 0, 'description': 'no'}
            ]
        },
        {
            'bit_number': 10,
            'qa_description': 'climatology source',
            'bit_value': [
                {'value': 1, 'description': 'climatology'},
                {'value': 0, 'description': 'MODIS'}
            ]
        },
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


# https://github.com/GERSL/Fmask#46-version
QA_SPECS.append(QA_ValueSpecTable({
    'qa_spec_name': 'FMASK',
    'qa_spec_date': '???',
    'sensor': '*',
    'dtype': {'kind': 'u', 'itemsize': 1},
    'values': [
        {'value': 0, 'qa_name': 'clear', 'qa_description': 'clear land pixel'},
        {'value': 1, 'qa_name': 'water', 'qa_description': 'clear water pixel'},
        {'value': 2, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow'},
        {'value': 3, 'qa_name': 'ice', 'qa_description': 'snow'},
        {'value': 4, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'value': 255, 'qa_description': 'no observation'}
    ]
}))


# https://github.com/GERSL/Fmask#46-version
QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'Phase1_QA',
    'qa_spec_date': '2022-03-28',
    'sensor': '*',
    'dtype': {'kind': 'u', 'itemsize': 1},
    'bits': [
        {'bit_number': 0, 'qa_name': 'combined', 'qa_description': 'TnE'},
        {'bit_number': 1, 'qa_name': 'dilated_cloud', 'qa_description': 'dilated_cloud'},
        {'bit_number': 2, 'qa_name': 'cirrus', 'qa_description': 'cirrus'},
        {'bit_number': 3, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'bit_number': 4, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud_shadow'},
        {'bit_number': 5, 'qa_name': 'ice', 'qa_description': 'snow'},
        {'bit_number': 6, 'qa_name': 'clear', 'qa_description': 'clear'},
        {'bit_number': 7, 'qa_name': 'water', 'qa_description': 'water'}
    ]
}))


# Sentinel2 L2A SCL scene classification mask
# https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
QA_SPECS.append(QA_ValueSpecTable({
    'qa_spec_name': 'SCL',
    'qa_spec_date': '???',
    'sensor': 'S2',
    'dtype': {'kind': 'u', 'itemsize': 1},
    'values': [
        {'value': 0, 'qa_name': 'nodata', 'qa_description': 'No data'},
        {'value': 1, 'qa_name': 'defective', 'qa_description': 'Saturated / Defective'},
        {'value': 2, 'qa_name': 'dark_area', 'qa_description': 'Dark Area Pixels'},
        {'value': 3, 'qa_name': 'cloud_shadow', 'qa_description': 'Cloud Shadows'},
        {'value': 4, 'qa_name': 'vegetation', 'qa_description': 'Vegetation'},
        {'value': 5, 'qa_name': 'bare', 'qa_description': 'Bare Soils'},
        {'value': 6, 'qa_name': 'water', 'qa_description': 'Water'},
        {'value': 7, 'qa_name': 'clouds_lo_prob', 'qa_description': 'Clouds low probability / Unclassified'},
        {'value': 8, 'qa_name': 'clouds_mid_prob', 'qa_description': 'Clouds medium probability'},
        {'value': 9, 'qa_name': 'cloud', 'qa_description': 'Clouds high probability'},
        {'value': 10, 'qa_name': 'cirrus', 'qa_description': 'Cirrus'},
        {'value': 11, 'qa_name': 'ice', 'qa_description': 'Snow / Ice'},
    ]
}))


# Landsat Level 2 QA bands
# https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
# https://www.usgs.gov/landsat-missions/landsat-collection-1-level-1-quality-assessment-band
QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'qa_pixel',
    'qa_spec_date': '???',
    'sensor': 'L8',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {'bit_number': 0, 'qa_name': 'fill', 'qa_description': 'Designated fill'},
        {'bit_number': 1, 'qa_name': 'terrain_occlusion', 'qa_description': 'Terrain Occlusion'},
        {'bit_number': 4, 'qa_name': 'cloud', 'qa_description': 'Cloud'},
    ],
    # Represents multi-valued groups
    'bit_ranges': [
        {'bit_range': [2, 3], 'qa_name': 'rad_sat', 'qa_description': 'Radiometric Saturation'},
        {'bit_range': [5, 6], 'qa_name': 'cloud_conf', 'qa_description': 'Cloud Confidence'},
        {'bit_range': [7, 8], 'qa_name': 'cloud_shadow_conf', 'qa_description': 'Cloud Shadow Confidence'},
        {'bit_range': [9, 10], 'qa_name': 'snow_conf', 'qa_description': 'Snow/Ice Confidence'},
        {'bit_range': [11, 12], 'qa_name': 'snow_conf', 'qa_description': 'Snow/Ice Confidence'},
    ]
}))


# ARA data
QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ARA-4',
    'qa_spec_date': '2024-03-06',
    'sensor': 'L8',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {
            'bit_number': 0,
            'qa_name': 'TnE',
            'qa_description': 'TnE Evaluate',
            'bit_value': [{'value': 1, 'description': 'use-pixel'}, {'value': 0, 'description': 'ignore-pixel'}]
        },
        {
            'bit_number': 1,
            'qa_name': 'dilated_cloud',
            'qa_description': 'cloud',
            'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]
        },
        {'bit_number': 2, 'qa_name': 'cirrus', 'qa_description': 'cirrus'},
        {'bit_number': 3, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'bit_number': 4, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow'},
        {'bit_number': 5, 'qa_name': 'snow', 'qa_description': 'snow'},
        {'bit_number': 6, 'qa_name': 'clear', 'qa_description': 'clear'},
        {'bit_number': 7, 'qa_name': 'water', 'qa_description': 'water'},
        {'bit_number': 8, 'qa_name': 'aligned', 'qa_description': 'image aligned'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ARA-4',
    'qa_spec_date': '2024-03-06',
    'sensor': 'S2',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {'bit_number': 0, 'qa_name': 'TnE', 'qa_description': 'TnE Evaluate', 'bit_value': [{'value': 1, 'description': 'use-pixel'}, {'value': 0, 'description': 'ignore-pixel'}]},
        {'bit_number': 2, 'qa_name': 'cirrus', 'qa_description': 'cirrus'},
        {'bit_number': 3, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'bit_number': 4, 'qa_name': 'cloud_shadow', 'qa_description': 'cloud shadow'},
        {'bit_number': 5, 'qa_name': 'snow', 'qa_description': 'snow'},
        {'bit_number': 6, 'qa_name': 'clear', 'qa_description': 'clear'},
        {'bit_number': 7, 'qa_name': 'water', 'qa_description': 'water'},
        {'bit_number': 8, 'qa_name': 'aligned', 'qa_description': 'image aligned'},
        {'bit_number': 10, 'qa_name': 'imputed', 'qa_description': 'cloud imputed'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ARA-4',
    'qa_spec_date': '2024-03-06',
    'sensor': 'PD',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {'bit_number': 0, 'qa_name': 'TnE', 'qa_description': 'TnE Evaluate', 'bit_value': [{'value': 1, 'description': 'use-pixel'}, {'value': 0, 'description': 'ignore-pixel'}]},
        {'bit_number': 1, 'qa_name': 'dilated_cloud', 'qa_description': 'cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 3, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'bit_number': 6, 'qa_name': 'clear', 'qa_description': 'clear'},
        {'bit_number': 8, 'qa_name': 'aligned', 'qa_description': 'image aligned'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


QA_SPECS.append(QA_BitSpecTable({
    'qa_spec_name': 'ARA-4',
    'qa_spec_date': '2024-03-06',
    'sensor': 'WV',
    'dtype': {'kind': 'u', 'itemsize': 2},
    'bits': [
        {'bit_number': 0, 'qa_name': 'TnE', 'qa_description': 'TnE Evaluate', 'bit_value': [{'value': 1, 'description': 'use-pixel'}, {'value': 0, 'description': 'ignore-pixel'}]},
        {'bit_number': 1, 'qa_name': 'dilated_cloud', 'qa_description': 'cloud', 'bit_value': [{'value': 1, 'description': 'yes'}, {'value': 0, 'description': 'no'}]},
        {'bit_number': 3, 'qa_name': 'cloud', 'qa_description': 'cloud'},
        {'bit_number': 3, 'qa_name': 'terrain_shadow', 'qa_description': 'Terrain Shadow'},
        {'bit_number': 5, 'qa_name': 'snow', 'qa_description': 'snow'},
        {'bit_number': 6, 'qa_name': 'clear', 'qa_description': 'clear'},
        {'bit_number': 7, 'qa_name': 'water', 'qa_description': 'water'},
        {'bit_number': 8, 'qa_name': 'aligned', 'qa_description': 'image aligned'},
        {'bit_number': 9, 'qa_name': 'orthorectified', 'qa_description': 'DSM Orthorectified'},
        {'bit_number': 11, 'qa_description': 'reserved for future use'},
        {'bit_number': 12, 'qa_description': 'reserved for future use'},
        {'bit_number': 13, 'qa_description': 'reserved for future use'},
        {'bit_number': 14, 'qa_description': 'reserved for future use'},
        {'bit_number': 15, 'qa_description': 'reserved for future use'}
    ]
}))


def demo():
    import sys
    fpath = sys.argv[1]
    from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS
    table = QA_SPECS.find_table('ACC-1', 'WV')
    import kwimage
    quality_im = kwimage.imread(fpath)
    drawings = table.draw_labels(quality_im)
    qa_canvas = drawings['qa_canvas']
    legend = drawings['legend']
    canvas = kwimage.stack_images([qa_canvas, legend], axis=1)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/tasks/fusion/datamodules/qa_bands.py /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop7-DEBUG/US_R007/WV/affine_warp/crop_20150401T160000Z_N34.190052W083.941277_N34.327136W083.776956_WV_0/crop_20150401T160000Z_N34.190052W083.941277_N34.327136W083.776956_WV_0_quality.tif
    """
    demo()
