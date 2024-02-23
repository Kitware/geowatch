import kwcoco
import kwimage
import numpy as np
import ubelt as ub
import einops


class BatchVisualizationBuilder:
    """
    Helper object to build a batch visualization.

    The basic logic is that we will build a column for each timestep and then
    arrange them from left to right to show how the scene changes over time.
    Each column will be made of "cells" which could show either the truth, a
    prediction, loss weights, or raw input channels.

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.datamodules.batch_visualization BatchVisualizationBuilder

    Example:
        >>> from geowatch.tasks.fusion.datamodules.batch_visualization import *  # NOQA
        >>> from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
        >>> import geowatch
        >>> coco_dset = geowatch.coerce_kwcoco('vidshapes2-geowatch', num_frames=5)
        >>> channels = 'r|g|b,B10|B8a|B1|B8|B11,X.2|Y.2'
        >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
        >>> # combinable_extra = None  # uncomment for raw behavior
        >>> self = KWCocoVideoDataset(
        >>>     coco_dset, time_dims=5, window_dims=(224, 256), channels=channels,
        >>>     use_centered_positives=True, neg_to_pos_ratio=0)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> item_output = BatchVisualizationBuilder.populate_demo_output(item, self.sampler.classes)
        >>> #binprobs[0][:] = 0  # first change prob should be all zeros
        >>> requested_tasks = self.requested_tasks
        >>> builder = BatchVisualizationBuilder(
        >>>     item, item_output, classes=self.classes, requested_tasks=requested_tasks,
        >>>     default_combinable_channels=self.default_combinable_channels, combinable_extra=combinable_extra)
        >>> #builder.overlay_on_image = 1
        >>> #canvas = builder.build()
        >>> builder.max_channels = 4
        >>> builder.overlay_on_image = 0
        >>> canvas2 = builder.build()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> #kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1))
        >>> #kwplot.imshow(canvas2, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.imshow(canvas2, fnum=1, doclf=True)
        >>> kwplot.show_if_requested()

    Example:
        >>> from geowatch.tasks.fusion.datamodules.batch_visualization import *  # NOQA
        >>> from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
        >>> import geowatch
        >>> coco_dset = geowatch.coerce_kwcoco('vidshapes2-geowatch', num_frames=5)
        >>> channels = 'r|g|b,B10|B8a|B1|B8|B11,X.2|Y.2'
        >>> #coco_dset = geowatch.coerce_kwcoco('vidshapes2', num_frames=5)
        >>> #channels = None
        >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
        >>> # combinable_extra = None  # uncomment for raw behavior
        >>> self = KWCocoVideoDataset(
        >>>     coco_dset, time_dims=5, window_dims=(128, 165), channels=channels,
        >>>     use_centered_positives=True, neg_to_pos_ratio=0, input_space_scale='native')
        >>> index = len(self) // 4
        >>> index = 0
        >>> target = native_target = self.new_sample_grid['targets'][index].copy()
        >>> #target['space_slice'] = (slice(224, 448), slice(224, 448))
        >>> target['space_slice'] = (slice(196, 196 + 148), slice(32, 128))
        >>> #target['space_slice'] = (slice(0, 196 + 148), slice(0, 128))
        >>> target['gids'] = target['gids']
        >>> #target['space_slice'] = (slice(16, 196 + 148), slice(16, 198))
        >>> #target['space_slice'] = (slice(-70, 196 + 148), slice(-128, 128))
        >>> native_target.pop('fliprot_params', None)
        >>> native_target['allow_augment'] = 0
        >>> native_item = self[native_target]
        >>> # Resample the same item, but without native scale sampling for comparison
        >>> rescaled_target = native_item['target'].copy()
        >>> rescaled_target.pop('fliprot_params', None)
        >>> rescaled_target['input_space_scale'] = 1
        >>> rescaled_target['output_space_scale'] = 1
        >>> rescaled_target['allow_augment'] = 0
        >>> rescale = 0
        >>> draw_weights = 1
        >>> rescaled_item = self[rescaled_target]
        >>> print(ub.urepr(self.summarize_item(native_item), nl=-1, sort=0))
        >>> native_item_output = BatchVisualizationBuilder.populate_demo_output(native_item, self.sampler.classes, rng=0)
        >>> rescaled_item_output = BatchVisualizationBuilder.populate_demo_output(rescaled_item, self.sampler.classes, rng=0)
        >>> #rescaled_item_output = None
        >>> #rescaled_item_output = None
        >>> #binprobs[0][:] = 0  # first change prob should be all zeros
        >>> requested_tasks = self.requested_tasks
        >>> builder = BatchVisualizationBuilder(
        >>>     native_item, native_item_output, classes=self.classes,
        >>>     requested_tasks=requested_tasks,
        >>>     default_combinable_channels=self.default_combinable_channels,
        >>>     combinable_extra=combinable_extra, rescale=rescale, draw_weights=draw_weights)
        >>> builder.max_channels = 4
        >>> builder.overlay_on_image = 0
        >>> native_canvas = builder.build()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> #kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1))
        >>> #kwplot.imshow(canvas2, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.imshow(native_canvas, fnum=1, doclf=True, figtitle='Native Sampling')
        >>> plt.gcf().tight_layout()
        >>> ######
        >>> # Resample the same item, but without native sampling for comparison
        >>> print(ub.urepr(self.summarize_item(rescaled_item), nl=-1, sort=0))
        >>> builder = BatchVisualizationBuilder(
        >>>     rescaled_item, rescaled_item_output, classes=self.classes,
        >>>     requested_tasks=requested_tasks,
        >>>     default_combinable_channels=self.default_combinable_channels,
        >>>     combinable_extra=combinable_extra, rescale=rescale, draw_weights=draw_weights)
        >>> builder.max_channels = 4
        >>> builder.overlay_on_image = 0
        >>> rescaled_canvas = builder.build()
        >>> kwplot.imshow(rescaled_canvas, fnum=2, doclf=True, figtitle='Rescaled Sampling')
        >>> plt.gcf().tight_layout()
        >>> ######
        >>> from geowatch.tasks.fusion.datamodules.batch_visualization import _debug_sample_in_context
        >>> _debug_sample_in_context(self, target)
        >>> kwplot.show_if_requested()
    """

    def __init__(builder, item, item_output=None, combinable_extra=None,
                 max_channels=5, max_dim=224, norm_over_time=0,
                 overlay_on_image=False, draw_weights=True,
                 draw_truth=True, classes=None,
                 default_combinable_channels=None, requested_tasks=None,
                 rescale=1):
        builder.max_channels = max_channels
        builder.max_dim = max_dim
        builder.norm_over_time = norm_over_time
        builder.combinable_extra = combinable_extra
        builder.item_output = item_output
        builder.item = item
        builder.overlay_on_image = overlay_on_image
        builder.draw_weights = draw_weights
        builder.draw_truth = draw_truth
        builder.requested_tasks = requested_tasks

        builder.classes = classes
        builder.default_combinable_channels = default_combinable_channels

        combinable_channels = default_combinable_channels
        if combinable_extra is not None:
            if isinstance(combinable_extra, str):
                # coerce combinable extra from a channel spec
                import kwcoco
                combinable_extra = [
                    s.to_oset() for s in kwcoco.ChannelSpec.coerce(combinable_extra).streams()]
            combinable_channels = combinable_channels.copy()
            combinable_channels += list(map(ub.oset, combinable_extra))
        builder.combinable_channels = combinable_channels
        builder.rescale = rescale

    @classmethod
    def populate_demo_output(cls, item, classes, rng=None):
        """
        Make dummy output for a batch item for testing
        """
        # Calculate the probability of change for each frame
        from geowatch.tasks.fusion.datamodules import data_utils
        import kwarray
        item_output = {}
        change_prob_list = []
        rng = kwarray.ensure_rng(rng)
        fliprot_params = item['target'].get('fliprot_params', None)
        for frame in item['frames'][1:]:  # first frame does not have change
            change_prob = kwimage.Heatmap.random(
                dims=frame['output_dims'], classes=1, rng=rng).data['class_probs'][0]
            if fliprot_params:
                change_prob = data_utils.fliprot(change_prob, **fliprot_params)
            change_prob_list += [change_prob]
        change_probs = change_prob_list
        item_output['change_probs'] = change_probs
        #
        # Probability of each class for each frame
        class_prob_list = []
        for frame in item['frames']:
            class_prob = kwimage.Heatmap.random(
                dims=frame['output_dims'], classes=list(classes), rng=rng).data['class_probs']
            class_prob = einops.rearrange(class_prob, 'c h w -> h w c')
            if fliprot_params:
                class_prob = data_utils.fliprot(class_prob, **fliprot_params)
            class_prob_list += [class_prob]
        class_probs = class_prob_list
        item_output['class_probs'] = class_probs
        #
        # Probability of "saliency" (i.e. non-background) for each frame
        saliency_prob_list = []
        for frame in item['frames']:
            saliency_prob = kwimage.Heatmap.random(
                dims=frame['output_dims'], classes=1, rng=rng).data['class_probs']
            saliency_prob = einops.rearrange(saliency_prob, 'c h w -> h w c')
            if fliprot_params:
                saliency_prob = data_utils.fliprot(saliency_prob, **fliprot_params)
            saliency_prob_list += [saliency_prob]
        saliency_probs = saliency_prob_list
        item_output['saliency_probs'] = saliency_probs

        #
        # Predicted bounding boxes for each frame
        pred_ltrb_list = []
        for frame in item['frames']:
            frame_output_dsize = frame['output_dims'][::-1]
            num_pred_boxes = rng.randint(0, 8)
            pred_boxes = kwimage.Boxes.random(num_pred_boxes).scale(frame_output_dsize)
            # if fliprot_params:
            #     ... = data_utils.fliprot_annot(saliency_prob, **fliprot_params)
            pred_ltrb_list.append(pred_boxes.to_ltrb().data)
        item_output['pred_ltrb'] = pred_ltrb_list
        return item_output

    def build(builder):
        frame_metas = builder._prepare_frame_metadata()
        if 0:
            for idx, frame_meta in enumerate(frame_metas):
                print('---')
                print('idx = {!r}'.format(idx))
                frame_weight_shape = ub.map_vals(lambda x: x.shape, frame_meta['frame_weight'])
                print('frame_weight_shape = {}'.format(ub.urepr(frame_weight_shape, nl=1)))
                frame_meta['frame_weight']
        canvas = builder._build_canvas(frame_metas)
        return canvas

    def _prepare_frame_metadata(builder):
        import more_itertools
        item = builder.item
        combinable_channels = builder.combinable_channels

        truth_keys = []
        weight_keys = []
        if builder.requested_tasks['class']:
            # TODO: prefer class-ohe if available
            truth_keys.append('class_idxs')
            weight_keys.append('class_weights')
        if builder.requested_tasks['saliency']:
            truth_keys.append('saliency')
            weight_keys.append('saliency_weights')
        if builder.requested_tasks['change']:
            truth_keys.append('change')
            weight_keys.append('change_weights')
        if builder.requested_tasks['outputs']:
            weight_keys.append('output_weights')

        # Prepare metadata on each frame
        frame_metas = []
        for frame_idx, frame_item in enumerate(item['frames']):
            # Gather ground truth rasters
            frame_truth = {}
            for truth_key in truth_keys:
                truth_data = frame_item[truth_key]
                if truth_data is not None:
                    truth_data = truth_data.data.cpu().numpy()
                    frame_truth[truth_key] = truth_data

            frame_weight = {}
            for weight_key in weight_keys:
                weight_data = frame_item[weight_key]
                if weight_data is not None:
                    weight_data = weight_data.data.cpu().numpy()
                    frame_weight[weight_key] = weight_data
                else:
                    # HACK so saliency weights align correctly
                    frame_weight[weight_key] = None
                    # np.full((2, 2), fill_value=np.nan)

            # Breakup all of the modes into 1-channel per array
            frame_chan_names = []
            frame_chan_datas = []
            frame_modes = frame_item['modes']
            for mode_code, mode_data in frame_modes.items():
                mode_data = mode_data.data.cpu().numpy()
                code_list = kwcoco.FusedChannelSpec.coerce(mode_code).normalize().as_list()
                for chan_data, chan_name in zip(mode_data, code_list):
                    frame_chan_names.append(chan_name)
                    frame_chan_datas.append(chan_data)
            full_mode_code = ','.join(list(frame_item['modes'].keys()))

            # Determine what single and combinable channels exist per stream
            perstream_available = []
            for mode_code in frame_modes.keys():
                code_list = kwcoco.FusedChannelSpec.coerce(mode_code).normalize().as_list()
                code_set = ub.oset(code_list)
                stream_combinables = []
                for combinable in combinable_channels:
                    if combinable.issubset(code_set):
                        stream_combinables.append(combinable)
                remain = code_set - set(ub.flatten(stream_combinables))
                stream_singletons = [(c,) for c in remain]
                # Prioritize combinable channels in each stream first
                stream_available = list(map(tuple, stream_combinables)) + stream_singletons
                perstream_available.append(stream_available)

            # Prioritize choosing a balance of channels from each stream
            frame_available_chans = list(more_itertools.roundrobin(*perstream_available))

            frame_meta = {
                'full_mode_code': full_mode_code,
                'frame_idx': frame_idx,
                'frame_item': frame_item,
                'frame_chan_names': frame_chan_names,
                'frame_chan_datas': frame_chan_datas,
                'frame_available_chans': frame_available_chans,
                'frame_truth': frame_truth,
                'frame_weight': frame_weight,
                'sensor': frame_item.get('sensor', '*'),
                ###
                'true_box_ltrb': frame_item.get('box_ltrb', None),
                'output_dims': frame_item.get('output_dims', None),
            }
            frame_metas.append(frame_meta)

        # Determine which frames to visualize For each frame choose N channels
        # such that common channels are aligned, visualize common channels in
        # the first rows and then fill with whatever is left
        # chan_freq = ub.dict_hist(ub.flatten(frame_meta['frame_available_chans']
        #                                     for frame_meta in frame_metas))
        # chan_priority = {k: (v, len(k), -idx) for idx, (k, v)
        #                  in enumerate(chan_freq.items())}
        for frame_meta in frame_metas:
            chan_keys = frame_meta['frame_available_chans']
            # print('chan_keys = {!r}'.format(chan_keys))
            # frame_priority = ub.dict_isect(chan_priority, chan_keys)
            # chosen = ub.argsort(frame_priority, reverse=True)[0:builder.max_channels]
            # print('chosen = {!r}'.format(chosen))
            chosen = chan_keys[0:builder.max_channels]
            frame_meta['chans_to_use'] = chosen

        # Gather channels to visualize
        for frame_meta in frame_metas:
            chans_to_use = frame_meta['chans_to_use']
            frame_chan_names = frame_meta['frame_chan_names']
            frame_chan_datas = frame_meta['frame_chan_datas']
            chan_idx_lut = {name: idx for idx, name in enumerate(frame_chan_names)}
            # Prepare and normalize the channels for visualization
            chan_rows = []
            for chan_names in chans_to_use:
                chan_code = '|'.join(chan_names)
                chanxs = list(ub.take(chan_idx_lut, chan_names))
                parts = list(ub.take(frame_chan_datas, chanxs))
                raw_signal = np.stack(parts, axis=2)
                row = {
                    'raw_signal': raw_signal,
                    'chan_code': chan_code,
                    'signal_text': f'{chan_code}',
                    'sensor': frame_meta['sensor'],
                }
                chan_rows.append(row)
            frame_meta['chan_rows'] = chan_rows
            assert len(chan_rows) > 0, 'no channels to draw on'

        if builder.draw_weights:
            # Normalize weights for visualization
            all_weight_overlays = []
            weight_shapes = []
            for frame_meta in frame_metas:
                frame_meta['weight_overlays'] = {}
                for weight_key, weight_data in frame_meta['frame_weight'].items():

                    if weight_data is not None:
                        weight_shapes.append(weight_data.shape)

                    overlay_row = {
                        'weight_key': weight_key,
                        'raw': weight_data,
                    }
                    frame_meta['weight_overlays'][weight_key] = overlay_row
                    all_weight_overlays.append(overlay_row)

            for weight_key, group in ub.group_items(all_weight_overlays, lambda x: x['weight_key']).items():
                for cell in group:
                    weight_data = cell['raw']
                    if weight_data is None:
                        if len(weight_shapes) == 0:
                            h = w = builder.max_dim
                        else:
                            h, w = weight_shapes[0]
                        weight_overlay = kwimage.draw_text_on_image(
                            {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                            valign='center', halign='center', fontScale=10,
                            color='kw_red')
                        weight_overlay = kwimage.ensure_float01(weight_overlay)
                    else:
                        # Normally weights will range between 0 and 1, but in
                        # some cases they may range higher.  We handle this by
                        # coloring the 0-1 range in grayscale and the
                        # 1-infinity range in color
                        weight_overlay = colorize_weights(weight_data)
                    cell['overlay'] = weight_overlay

        # Normalize raw signal into visualizable range
        if builder.norm_over_time:
            # Normalize all cells with the same channel code across time
            channel_cells = [cell for frame_meta in frame_metas for cell in frame_meta['chan_rows']]
            # chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'])
            chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'], c['sensor']))
            for chan_code, cells in chan_to_cells.items():
                flat = [c['raw_signal'].ravel() for c in cells]
                cums = np.cumsum(list(map(len, flat)))
                combo = np.hstack(flat)
                mask = (combo != 0) & np.isfinite(combo)
                # try:
                combo_normed = kwimage.normalize_intensity(combo, mask=mask).copy()
                # except Exception:
                #     combo_normed = combo.copy()
                flat_normed = np.split(combo_normed, cums)
                for cell, flat_item in zip(cells, flat_normed):
                    norm_signal = flat_item.reshape(*cell['raw_signal'].shape)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    # norm_signal = np.nan_to_num(norm_signal)
                    norm_signal = kwimage.fill_nans_with_checkers(norm_signal)
                    cell['norm_signal'] = norm_signal
        else:
            import warnings
            from geowatch.utils import util_kwimage
            # Normalize each timestep by itself
            for frame_meta in frame_metas:
                for row in frame_meta['chan_rows']:
                    raw_signal = row['raw_signal']

                    # HACK:
                    # There are certain bands that are integral label images When they are
                    # drawn by themselves we can colorize them.  It would be nice to make the
                    # labeling consistent, but this is probably better than pure grayscale.
                    LABEL_CHANNELS = {'quality', 'cloudmask'}
                    is_label_img = row['chan_code'] in LABEL_CHANNELS

                    if is_label_img:
                        needs_norm = False
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='All-NaN slice')
                            if raw_signal.dtype.kind == 'u' and raw_signal.dtype.itemsize == 1:
                                raw_signal = kwimage.ensure_float01(raw_signal)
                                needs_norm = False
                            else:
                                try:
                                    needs_norm = np.nanmin(raw_signal) < 0 or np.nanmax(raw_signal) > 1
                                except Exception:
                                    needs_norm = False

                    if needs_norm:
                        mask = (raw_signal != 0) & np.isfinite(raw_signal)
                        norm_signal = kwimage.normalize_intensity(raw_signal, mask=mask, params={'scaling': 'sigmoid'}).copy()
                    elif is_label_img:
                        raw_signal = util_kwimage.exactly_1channel(raw_signal, ndim=2)
                        norm_signal = util_kwimage.colorize_label_image(raw_signal, with_legend=False)
                    else:
                        norm_signal = raw_signal.copy()

                    norm_signal = kwimage.fill_nans_with_checkers(norm_signal)
                    norm_signal = util_kwimage.ensure_false_color(norm_signal)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    row['norm_signal'] = norm_signal

        return frame_metas

    def _build_canvas(builder, frame_metas):

        # Given prepared frame metadata, build a vertical stack of per-chanel
        # information, and then horizontally stack the timesteps.
        horizontal_stack = []

        truth_overlay_keys = set(ub.flatten([m['frame_truth'] for m in frame_metas]))
        weight_overlay_keys = set(ub.flatten([m['frame_weight'] for m in frame_metas]))

        vertical_stacks = []
        for frame_meta in frame_metas:
            vertical_stack = builder._build_frame_vertical_stack(
                frame_meta, truth_overlay_keys, weight_overlay_keys)
            vertical_stacks.append(vertical_stack)

        # Make the headers the same height in each stack
        for row_stack in zip(*vertical_stacks):
            if all(r['type'] == 'header' for r in row_stack) :
                heights = [r['im'].shape[0] for r in row_stack]
                if not ub.allsame(heights):
                    max_h = max(heights)
                    for r in row_stack:
                        h, w = r['im'].shape[0:2]
                        if h != max_h:
                            r['im'] = kwimage.imresize(r['im'], dsize=(w, max_h), letterbox=True)

        if 0:
            stack_shape_texts = []
            for vertical_stack in vertical_stacks:
                text = '\n'.join([str(r['im'].shape) for r in vertical_stack])
                stack_shape_texts.append(text)
            print(ub.hzcat(stack_shape_texts))

        for vertical_stack in vertical_stacks:
            frame_canvas = kwimage.stack_images([r['im'] for r in vertical_stack], pad=3, bg_value='kitware_darkgreen')
            horizontal_stack.append(frame_canvas)

        body_canvas = kwimage.stack_images(horizontal_stack, axis=1, pad=5, bg_value='kitware_darkblue')
        body_canvas = body_canvas[..., 0:3]  # drop alpha
        body_canvas = kwimage.ensure_uint255(body_canvas)  # convert to uint8

        width = body_canvas.shape[1]

        vid_text = f'video: {builder.item["video_id"]} - {builder.item["video_name"]}'

        # producer_rank = builder.item.get('producer_rank', None)
        # producer_mode = builder.item.get('producer_mode', None)
        # requested_index = builder.item.get('requested_index', None)
        # resolved_index = builder.item.get('resolved_index', None)
        # if producer_rank is not None:
        #     vid_text += f'\nrank={producer_rank} {producer_mode} {requested_index=} {resolved_index=}'

        sample_gsd = builder.item.get('sample_gsd', None)
        if sample_gsd is not None:
            if isinstance(sample_gsd, float):
                vid_text = vid_text + ' @ {:0.2f} GSD'.format(sample_gsd)
            else:
                vid_text = vid_text + ' @ {} GSD'.format(sample_gsd)

        vid_header = kwimage.draw_text_on_image(
            {'width': width}, vid_text, org=(width // 2, 3), valign='top',
            halign='center', color='pink')

        canvas = kwimage.stack_images([vid_header, body_canvas], axis=0, pad=3, bg_value='kitware_darkblue')
        return canvas

    def _build_frame_header(builder, frame_meta):
        """
        Make the text header for each timestep (frame)
        """
        header_stack = []

        frame_item = frame_meta['frame_item']
        frame_idx = frame_meta['frame_idx']
        gid = frame_item['gid']

        # Build column headers
        header_dims = {'width': builder.max_dim}
        header_part = kwimage.draw_header_text(
            image=header_dims, fit=False,
            text=f't={frame_idx} gid={gid}', color='salmon')
        header_stack.append({
            'im': header_part,
            'type': 'header',
        })

        sensor = frame_item.get('sensor', '*')
        if sensor != '*':
            header_part = kwimage.draw_header_text(
                image=header_dims, fit=False, text=f'{sensor}',
                color='salmon')
            header_stack.append({
                'im': header_part,
                'type': 'header',
            })

        date_captured = frame_item.get('date_captured', '')
        if date_captured:
            header_part = kwimage.draw_header_text(
                header_dims, fit='shrink', text=f'{date_captured}',
                color='salmon')
            header_stack.append({
                'im': header_part,
                'type': 'header',
            })
        return header_stack

    def _build_frame_vertical_stack(builder, frame_meta, truth_overlay_keys, weight_overlay_keys):
        """
        Build a vertical stack for a single frame
        """
        classes = builder.classes
        item_output = builder.item_output

        vertical_stack = []

        frame_idx = frame_meta['frame_idx']
        chan_rows = frame_meta['chan_rows']

        frame_truth = frame_meta['frame_truth']
        # frame_weight = frame_meta['frame_weight']

        # Build column headers
        header_stack = builder._build_frame_header(frame_meta)
        vertical_stack.extend(header_stack)

        overlay_shape = tuple(frame_meta['output_dims'])
        if overlay_shape is None:
            overlay_shape = (32, 32)
        # # Build truth / metadata overlays
        # if len(frame_truth):
        #     overlay_shape = ub.peek(frame_truth.values()).shape[0:2]
        # else:
        #     overlay_shape = None

        # Create overlays for training objective targets
        overlay_items = []

        true_box_ltrb = frame_meta.get('true_box_ltrb', None)
        if true_box_ltrb is not None:
            true_boxes = kwimage.Boxes(true_box_ltrb, 'ltrb').numpy()
        else:
            true_boxes = None

        if builder.draw_truth:
            # Create the true class label overlay
            # TODO: prefer class-ohe if available
            overlay_key = 'class_idxs'
            if overlay_key in truth_overlay_keys and builder.requested_tasks['class']:
                class_idxs = frame_truth.get(overlay_key, None)
                true_heatmap = kwimage.Heatmap(class_idx=class_idxs, classes=classes)
                overlay = true_heatmap.colorize('class_idx')
                overlay[..., 3] = 0.5
                overlay_items.append({
                    'overlay': overlay,
                    'label_text': 'true class',
                })

            # Create the true saliency label overlay
            overlay_key = 'saliency'
            if overlay_key in truth_overlay_keys and builder.requested_tasks['saliency']:
                saliency = frame_truth.get(overlay_key, None)
                if saliency is not None:
                    if 1:
                        overlay = kwimage.make_heatmask(saliency.astype(np.float32), cmap='plasma').clip(0, 1)
                        overlay[..., 3] *= 0.5
                    else:
                        overlay = np.zeros(saliency.shape + (4,), dtype=np.float32)
                        overlay = kwimage.Mask(saliency, format='c_mask').draw_on(overlay, color='dodgerblue')
                        overlay = kwimage.ensure_alpha_channel(overlay)
                        overlay[..., 3] = (saliency > 0).astype(np.float32) * 0.5
                    overlay_items.append({
                        'overlay': overlay,
                        'label_text': 'true saliency',
                    })

            # Create the true change label overlay
            overlay_key = 'change'
            if overlay_key in truth_overlay_keys and builder.requested_tasks['change']:
                overlay = np.zeros(overlay_shape + (4,), dtype=np.float32)
                changes = frame_truth.get(overlay_key, None)
                if changes is not None:
                    if 1:
                        overlay = kwimage.make_heatmask(changes.astype(np.float32), cmap='viridis').clip(0, 1)
                        overlay[..., 3] *= 0.5
                    else:
                        overlay = kwimage.Mask(changes, format='c_mask').draw_on(overlay, color='lime')
                        overlay = kwimage.ensure_alpha_channel(overlay)
                        overlay[..., 3] = (changes > 0).astype(np.float32) * 0.5
                overlay_items.append({
                    'overlay': overlay,
                    'label_text': 'true change',
                })

            overlay_key = 'true_box_ltrb'
            # if overlay_key in truth_overlay_keys and builder.requested_tasks['boxes']:
            if true_boxes is not None and builder.requested_tasks['boxes']:
                overlay = np.zeros(overlay_shape + (4,), dtype=np.float32)
                dim = max(*overlay_shape)
                thickness = max(1, int(dim // 64))
                if true_boxes is not None:
                    overlay = true_boxes.draw_on(overlay, color='kitware_green', thickness=thickness)
                overlay_items.append({
                    'overlay': overlay,
                    'label_text': 'true boxes',
                })

        weight_items = []
        if builder.draw_weights:
            weight_overlays = frame_meta['weight_overlays']
            for overlay_key in weight_overlay_keys:
                weight_overlay_info = weight_overlays.get(overlay_key, None)
                if weight_overlay_info is not None:
                    weight_items.append({
                        'overlay': weight_overlay_info['overlay'],
                        'label_text': overlay_key,
                    })

        resizekw = {
            'dsize': (builder.max_dim, builder.max_dim),
            # 'max_dim': builder.max_dim,
            # 'letterbox': False,
            'letterbox': True,
            'interpolation': 'nearest',
            'border_value': 'kitware_darkgray',
            # 'interpolation': 'linear',
        }

        # TODO: clean up logic

        key = 'class_probs'
        overlay_index = 0
        if item_output and key in item_output and builder.requested_tasks['class']:
            x = item_output[key][frame_idx]
            x_shape = x.shape[0:2]
            if builder.overlay_on_image:
                norm_signal = chan_rows[overlay_index]['norm_signal']
                norm_signal = kwimage.imresize(norm_signal, dsize=x_shape[::-1])
            else:
                norm_signal = np.zeros(x_shape + (3,), dtype=np.float32)
            class_probs = einops.rearrange(x, 'h w c -> c h w')
            class_heatmap = kwimage.Heatmap(class_probs=class_probs, classes=classes)
            pred_part = class_heatmap.draw_on(norm_signal, with_alpha=0.7)
            # TODO: we might want to overlay the prediction on one or
            # all of the channels

            if builder.rescale:
                pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)

            pred_text = f'pred class t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='dodgerblue', border=3)
            vertical_stack.append({
                'im': pred_part,
                'type': 'data',
            })

        key = 'saliency_probs'
        if item_output and key in item_output and builder.requested_tasks['saliency']:
            x = item_output[key][frame_idx]
            x_shape = x.shape[0:2]
            if builder.overlay_on_image:
                norm_signal = chan_rows[0]['norm_signal']
                norm_signal = kwimage.imresize(norm_signal, dsize=x_shape[::-1])
            else:
                norm_signal = np.zeros(x_shape + (3,), dtype=np.float32)
            saliency_probs = einops.rearrange(x, 'h w c -> c h w')
            # Hard coded index, dont like
            is_salient_probs = saliency_probs[1]
            # saliency_heatmap = kwimage.Heatmap(class_probs=saliency_probs)
            # pred_part = saliency_heatmap.draw_on(norm_signal, with_alpha=0.7)
            pred_part = kwimage.make_heatmask(is_salient_probs, cmap='plasma')
            pred_part[..., 3] = 0.7
            # TODO: we might want to overlay the prediction on one or
            # all of the channels

            if builder.rescale:
                pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)

            pred_text = f'pred saliency t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='dodgerblue', border=3)
            vertical_stack.append({
                'im': pred_part,
                'type': 'data',
            })

        key = 'change_probs'
        overlay_index = 1
        if item_output and key in item_output and builder.requested_tasks['change']:
            # Make a probability heatmap we can either display
            # independently or overlay on a rendered channel
            if frame_idx == 0:
                # BIG RED X
                # h, w = vertical_stack[-1].shape[0:2]
                h = w = builder.max_dim
                pred_mask = kwimage.draw_text_on_image(
                    {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                    valign='center', halign='center', fontScale=10,
                    color='red')
                pred_part = pred_mask
            else:
                x = item_output[key][frame_idx - 1]
                x_shape = x.shape[0:2]
                # Draw predictions on the first item
                pred_mask = kwimage.make_heatmask(x, cmap='viridis')
                norm_signal = chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal']
                if builder.overlay_on_image:
                    norm_signal = norm_signal
                    norm_signal = kwimage.imresize(norm_signal, dsize=x_shape[::-1])
                else:
                    norm_signal = np.zeros(x_shape + (3,), dtype=np.float32)
                pred_layers = [pred_mask, norm_signal]
                pred_part = kwimage.overlay_alpha_layers(pred_layers)
                # TODO: we might want to overlay the prediction on one or
                # all of the channels

                if builder.rescale:
                    pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)

                pred_text = f'pred change t={frame_idx}'
                pred_part = kwimage.draw_text_on_image(
                    pred_part, pred_text, (1, 1), valign='top',
                    color='dodgerblue', border=3)
            vertical_stack.append({
                'im': pred_part,
                'type': 'data',
            })

        key = 'pred_ltrb'
        overlay_index = 0
        if item_output and key in item_output and builder.requested_tasks['boxes']:
            pred_ltrb = item_output[key][frame_idx]
            pred_boxes = kwimage.Boxes(pred_ltrb, 'ltrb')
            x_shape = overlay_shape
            if builder.overlay_on_image:
                norm_signal = chan_rows[overlay_index]['norm_signal']
                norm_signal = kwimage.imresize(norm_signal, dsize=x_shape[::-1])
            else:
                norm_signal = np.zeros(x_shape + (3,), dtype=np.float32)
            pred_part = pred_boxes.draw_on(norm_signal, alpha=0.7,
                                           color='kitware_blue', thickness=16)
            if builder.rescale:
                pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)

            pred_text = f'pred boxes t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='kitware_blue', border=3)
            vertical_stack.append({
                'im': pred_part,
                'type': 'data',
            })

        if not builder.overlay_on_image:
            # FIXME: might be broken
            # Draw the overlays by themselves
            for overlay_info in overlay_items:
                _draw_overlay_item_by_itself(builder, overlay_info, resizekw)
                stack_item = _draw_overlay_item_by_itself(
                    builder, overlay_info, resizekw)
                vertical_stack.append(stack_item)

        for overlay_info in weight_items:
            stack_item = _draw_overlay_item_by_itself(
                builder, overlay_info, resizekw)
            vertical_stack.append(stack_item)

        iterx = -1
        for iterx, row in enumerate(chan_rows):

            overlay_info = None
            if builder.overlay_on_image:
                # Request an overlay on top of this item
                if iterx < len(overlay_items):
                    overlay_info = overlay_items[iterx]

            stack_item = _draw_row_item(
                row, builder, overlay_info, resizekw)
            vertical_stack.append(stack_item)

        # If there aren't enough data items to draw the overlay on, then
        # add more...
        if builder.overlay_on_image:
            if iterx < len(overlay_items):
                pass

        for row in vertical_stack:
            row['im'] = kwimage.ensure_uint255(row['im'])
        return vertical_stack


def _draw_overlay_item_by_itself(builder, overlay_info, resizekw):
    label_text = overlay_info['label_text']
    row_canvas = overlay_info['overlay'][..., 0:3]

    if builder.rescale:
        row_canvas = kwimage.imresize(row_canvas, **resizekw)

    row_canvas = row_canvas.clip(0, 1)
    signal_bottom_y = 1  # hack: hardcoded
    row_canvas = kwimage.ensure_uint255(row_canvas)
    row_canvas = kwimage.draw_text_on_image(
        row_canvas, label_text, (1, signal_bottom_y + 1),
        valign='top', color='lime', border=3)
    stack_item = {
        'im': row_canvas,
        'type': 'data',
    }
    return stack_item


def _draw_row_item(row, builder, overlay_info, resizekw):
    layers = []
    label_text = None
    norm_signal = row['norm_signal']
    if overlay_info is not None:
        # Draw truth on the image itself
        overlay = overlay_info['overlay']
        overlay = kwimage.imresize(
            overlay, dsize=norm_signal.shape[0:2][::-1])
        layers.append(overlay)
        label_text = overlay_info['label_text']

    layers.append(norm_signal)
    row_canvas = kwimage.overlay_alpha_layers(layers)[..., 0:3]

    if builder.rescale:
        row_canvas = kwimage.imresize(row_canvas, **resizekw)

    row_canvas = row_canvas.clip(0, 1)
    row_canvas = kwimage.ensure_uint255(row_canvas)
    row_canvas = kwimage.draw_text_on_image(
        row_canvas, row['signal_text'], (1, 1), valign='top',
        color='white', border=3)

    if label_text:
        # TODO: make draw_text_on_image able to return the
        # geometry of what it drew and use that.
        signal_bottom_y = 31  # hack: hardcoded
        row_canvas = kwimage.draw_text_on_image(
            row_canvas, label_text, (1, signal_bottom_y + 1),
            valign='top', color='lime', border=3)
    stack_item = {
        'im': row_canvas,
        'type': 'data',
    }
    return stack_item


def _debug_sample_in_context(self, target):
    """
    Draw the sampled images in videospace and draw the sample box on top of it
    so we can check to ensure the sampled data corresponds.

    This would be a nice helper for ndsampler itself (or at least the dataset).
    """

    coco_dset = self.sampler.dset
    coco_images = coco_dset.images(target['gids']).coco_images

    import kwplot
    plt = kwplot.autoplt()

    canvas_sequence = []
    vidspace_boxes = []

    for coco_img in coco_images:
        sensor = coco_img.img.get('sensor_coarse', '*')
        img_channels = self.input_sensorchan.matching_sensor(sensor).chans
        three_chans = img_channels.fuse().to_list()[0:3]
        if len(three_chans) == 2:
            three_chans = three_chans[0:1]
        delayed = coco_img.imdelay(channels=three_chans, space='video')
        vidspace_img = delayed.finalize()
        if vidspace_img.dtype.kind == 'u' and vidspace_img.dtype.itemsize == 1:
            vispace_canvas = kwimage.ensure_float01(vidspace_img.copy())
        else:
            vispace_canvas = kwimage.normalize_intensity(vidspace_img, axis=2)
        # vispace_canvas = np.ascontiguousarray(vispace_canvas)

        imgspace_frame_dets = coco_dset.annots(gid=coco_img.img['id']).detections
        vidspace_frame_dets = imgspace_frame_dets.warp(coco_img.warp_vid_from_img)

        sample_box = kwimage.Boxes.from_slice(target['space_slice'], clip=False, wrap=False)
        vispace_canvas = vidspace_frame_dets.draw_on(vispace_canvas)
        # vispace_canvas = sample_box.draw_on(vispace_canvas, color='kitware_orange', thickness=10)
        vidspace_boxes.append(sample_box)
        canvas_sequence.append(vispace_canvas)

    sequence_canvas, info = kwimage.stack_images(canvas_sequence, axis=1, pad=50, return_info=True)

    # kwimage
    kwplot.imshow(sequence_canvas, fnum=3, doclf=1)
    ax = plt.gca()
    ax.set_clip_on(False)

    for box, tf in zip(vidspace_boxes, info):
        box = box.warp(tf)
        print(f'box={box}')
        box.draw(color='kitware_orange', lw=4, alpha=0.8, ax=ax)
    ax.set_title('Sample Window in Video Space')


def colorize_weights(weights):
    """
    Normally weights will range between 0 and 1, but in some cases they may
    range higher.  We handle this by coloring the 0-1 range in grayscale and
    the 1-infinity range in color

    Example:
        >>> from geowatch.tasks.fusion.datamodules.batch_visualization import *  # NOQA
        >>> import kwarray
        >>> weights = kwimage.gaussian_patch((32, 32))
        >>> weights = kwarray.normalize(weights)
        >>> weights[:16, :16] *= 10
        >>> weights[16:, :16] *= 100
        >>> weights[16:, 16:] *= 1000
        >>> weights[:16, 16:] *= 10000
        >>> canvas = colorize_weights(weights)
        >>> # xdoctest: +REQUIRES(--show)
        >>> canvas = kwimage.imresize(canvas, dsize=(512, 512), interpolation='nearest').clip(0, 1)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-10', org=(1, 1), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-100', org=(256, 1), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-1000', org=(256, 256), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-10000', org=(1, 256), border=True)
        >>> import kwplot
        >>> import kwplot
        >>> kwplot.plt.ion()
        >>> kwplot.imshow(canvas)

    Example:
        >>> from geowatch.tasks.fusion.datamodules.batch_visualization import *  # NOQA
        >>> import kwarray
        >>> weights = kwimage.gaussian_patch((32, 32))
        >>> n = 512
        >>> weight_rows = [
        >>>     np.linspace(0, 1, n),
        >>>     np.linspace(0, 10, n),
        >>>     np.linspace(0, 100, n),
        >>>     np.linspace(0, 1000, n),
        >>>     np.linspace(0, 2000, n),
        >>>     np.linspace(0, 5000, n),
        >>>     np.linspace(0, 8000, n),
        >>>     np.linspace(0, 10000, n),
        >>>     np.linspace(0, 100000, n),
        >>>     np.linspace(0, 1000000, n),
        >>> ]
        >>> canvas = np.array([colorize_weights(row[None, :])[0] for row in weight_rows])
        >>> # xdoctest: +REQUIRES(--show)
        >>> canvas = kwimage.imresize(canvas, dsize=(512, 512), interpolation='nearest').clip(0, 1)
        >>> p = int(512 / len(weight_rows))
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-1', org=(1, 1 + p * 0), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-10', org=(1, 1 + p * 1), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-100', org=(1, 1 + p * 2), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-1000', org=(1, 1 + p * 3), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-2000', org=(1, 1 + p * 4), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-5000', org=(1, 1 + p * 5), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-8000', org=(1, 1 + p * 6), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-10000', org=(1, 1 + p * 7), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-100000', org=(1, 1 + p * 8), border=True)
        >>> canvas = kwimage.draw_text_on_image(canvas, '0-1000000', org=(1, 1 + p * 9), border=True)
        >>> import kwplot
        >>> import kwplot
        >>> kwplot.plt.ion()
        >>> kwplot.imshow(canvas)
    """
    canvas = kwimage.atleast_3channels(weights.copy())
    is_gt_one = weights > 1.0
    if np.any(is_gt_one):
        import matplotlib as mpl
        import matplotlib.cm  # NOQA
        from scipy import interpolate

        cmap_ = mpl.colormaps['YlOrRd']
        # cmap_ = mpl.colormaps['gist_rainbow']

        gt_one_values = weights[is_gt_one]

        max_val = gt_one_values.max()
        # Define a function that maps values from [1,inf) to [0,1]
        # the last max value part does depend on the inputs, which is fine.
        mapper = interpolate.interp1d(x=[1.0, 10.0, 100.0, max(max_val, 1000.0)],
                                      y=[0.0, 0.5, 0.75, 1.0])
        cmap_values = mapper(gt_one_values)
        colors01 = cmap_(cmap_values)[..., 0:3]

        rs, cs = np.where(is_gt_one)
        canvas[rs, cs, :] = colors01
    return canvas
