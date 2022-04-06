#!/usr/bin/env python
import json
import kwarray
import kwcoco
import kwimage
import numpy as np
import os
import pandas as pd
import sklearn.metrics as skm
import ubelt as ub
import warnings
from watch.tasks.fusion import utils
# from watch.utils import util_kwimage
from watch.utils import kwcoco_extensions
from kwcoco.coco_evaluator import CocoSingleResult
from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
from kwcoco.metrics.confusion_measures import OneVersusRestMeasureCombiner
from kwcoco.metrics.confusion_vectors import OneVsRestConfusionVectors
from kwcoco.metrics.confusion_measures import MeasureCombiner
# from kwcoco.metrics.confusion_measures import PerClass_Measures
from kwcoco.metrics.confusion_measures import Measures
from typing import Dict
import scriptconfig as scfg  # NOQA

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class SegmentationEvalConfig(scfg.Config):
    """
    Evaluation script for change/segmentation task
    """
    default = {
         'true_dataset': scfg.Value(None, help='path to the groundtruth dataset'),
         'pred_dataset': scfg.Value(None, help='path to the predicted dataset'),
         'eval_dpath': scfg.Value(None, help='path to dump results'),
         'draw_curves': scfg.Value('auto', help='flag to draw curves or not'),
         'draw_heatmaps': scfg.Value('auto', help='flag to draw heatmaps or not'),
         'score_space': scfg.Value('video', help='can score in image or video space'),
         'workers': scfg.Value('auto', help='number of parallel scoring workers'),
         'draw_workers': scfg.Value('auto', help='number of parallel drawing workers'),
    }


# def make_evaluate_config(cmdline=False, **kwargs):
#     from watch.utils.configargparse_ext import ArgumentParser
#     parser = ArgumentParser(
#         add_config_file_help=False,
#         description='Evaluation script for change/segmentation task',
#         auto_env_var_prefix='WATCH_FUSION_EVAL_',
#         add_env_var_help=True,
#         formatter_class='defaults',
#         config_file_parser_class='yaml',
#         args_for_setting_config_path=['--config'],
#         args_for_writing_out_config_file=['--dump'],
#     )
#     parser.add_argument('--true_dataset', '--test_dataset', help='path to the groundtruth dataset')
#     parser.add_argument('--pred_dataset', help='path to the predicted dataset')
#     parser.add_argument('--eval_dpath', help='path to dump results')
#     parser.add_argument('--draw_curves', default='auto', help='flag to draw curves or not')
#     parser.add_argument('--draw_heatmaps', default='auto', help='flag to draw heatmaps or not')
#     parser.add_argument('--score_space', default='video', help='can score in image or video space')
#     parser.add_argument('--workers', default='auto', help='number of parallel scoring workers')
#     parser.add_argument('--draw_workers', default='auto', help='number of parallel drawing workers')
#     parser.set_defaults(**kwargs)
#     default_args = None if cmdline else []
#     args, _ = parser.parse_known_args(default_args)
#     return args


def main(cmdline=True, **kwargs):
    """
    Ignore:
        from watch.tasks.fusion.evaluate import *  # NOQA
        from watch.tasks.fusion.evaluate import _memo_legend, _redraw_measures
        kwargs = {
            'config': './debug-eval.yaml',
            'draw_heatmaps': 0,
            'draw_curves': 0,
            'workers': 2,
            'score_space': 'video',
        }
        cmdline = False
    """

    # args = make_evaluate_config(cmdline=cmdline, **kwargs)
    # config = args.__dict__

    cfgpath = kwargs.pop('config', None)
    config = SegmentationEvalConfig(cmdline=cmdline, data=kwargs)
    if cfgpath is not None:
        # Hack, scriptconfig should have a nicer API than this
        config = config.load(cfgpath)
    config.update(kwargs)

    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    true_coco = kwcoco.CocoDataset.coerce(config['true_dataset'])
    pred_fpath = config['pred_dataset']

    try:
        pred_coco = kwcoco.CocoDataset.coerce(pred_fpath)
    except Exception:
        # Hack around issue in coerce
        pred_coco = kwcoco.CocoDataset()
        pred_coco.fpath = pred_fpath

    workers = config['workers']
    score_space = config['score_space']
    draw_workers = config['workers']
    eval_dpath = config['eval_dpath']

    from scriptconfig.smartcast import smartcast
    draw_heatmaps = smartcast(config['draw_heatmaps'])
    draw_curves = smartcast(config['draw_curves'])
    evaluate_segmentations(true_coco, pred_coco, eval_dpath,
                           draw_curves=draw_curves,
                           draw_heatmaps=draw_heatmaps,
                           score_space=score_space, workers=workers,
                           draw_workers=draw_workers)


@profile
def single_image_segmentation_metrics(pred_coco_img, true_coco_img,
                                      true_classes, true_dets, video1=None,
                                      score_space='video', thresh_bins=None):
    """
    Args:
        true_coco_img (kwcoco.CocoImage): detatched true coco image

        pred_coco_img (kwcoco.CocoImage): detatched predicted coco image

        thresh_bins (int): if specified rounds scores into this many bins
            to make calculating metrics more efficient
    """
    true_gid = true_coco_img.img['id']
    pred_gid = pred_coco_img.img['id']

    if thresh_bins is not None:
        if isinstance(thresh_bins, int):
            left_bin_edges = np.linspace(0, 1, thresh_bins)
        else:
            left_bin_edges = thresh_bins
    else:
        left_bin_edges = None

    img1 = true_coco_img.img

    if score_space == 'image':
        shape = (img1['height'], img1['width'])
    elif score_space == 'video':
        shape = (video1['height'], video1['width'])
    else:
        raise KeyError(score_space)

    row = {
        'true_gid': true_gid,
        'pred_gid': pred_gid,
    }
    if video1 is not None:
        row['video'] = video1['name']

    info = {
        'row': row,
        'shape': shape,
    }

    # TODO: parametarize these class categories
    from watch import heuristics
    ignore_classes = heuristics.IGNORE_CLASSNAMES
    background_classes = heuristics.BACKGROUND_CLASSES
    undistinguished_classes = heuristics.UNDISTINGUISHED_CLASSES
    context_classes = heuristics.CONTEXT_CLASSES
    negative_classes = heuristics.NEGATIVE_CLASSES
    # HACK! FIXME: There needs to be a clear definition of what classes are
    # scored and which are not.
    background_classes = background_classes | negative_classes
    """
    The above heuristics should roughtly be:

        * ignore_classes - ignore, Unknown
        * background_classes - background, negative
        * undistinguished_classes - positive
        * context_classes - No Activity Post Construction

        inferred:

        * class_scored_classes - Site Preperation, Active Construction
        * salient_scored_classes - positive, Site Preperation, Active Construction
    """

    # Determine what true/predicted categories are in common
    predicted_classes = []
    for stream in pred_coco_img.channels.streams():
        have = stream.intersection(true_classes)
        predicted_classes.extend(have.parsed)

    classes_of_interest = ub.oset(predicted_classes) - (
        negative_classes | background_classes | ignore_classes |
        undistinguished_classes)

    # Determine if saliency has been predicted
    salient_class = 'salient'
    has_saliency = salient_class in pred_coco_img.channels

    # Load ground truth annotations
    if score_space == 'video':
        warp_img_to_vid = kwimage.Affine.coerce(
            true_coco_img.img['warp_img_to_vid'])
        true_dets = true_dets.warp(warp_img_to_vid)
    true_cidxs = true_dets.data['class_idxs']
    true_ssegs = true_dets.data['segmentations']
    true_catnames = list(ub.take(true_dets.classes.idx_to_node, true_cidxs))
    info['true_dets'] = true_dets

    # NOTE: The exact definition of how we build the "truth" segmentation mask
    # is up for debate. I think this is a reasonable definition, but this needs
    # to be reviewed. It also likely needs updating to become general and
    # remove the need for heuristics.

    # We might need to:
    #     * add in a per-category weight canvas. This lets us say we can ignore
    #     clas A when scoring class B. Is there an example where this is
    #     relevant?

    # Does negative get moved to the background or scored?
    # Currently I'm just moving it to the background

    # How do we distinguish that

    # TODO:
    # Use the "valid_polygon" to zero out evaluations in invalid regions
    # Also use nan values in the predictions to do the same.
    # Combine these two measures.

    # Create a truth "panoptic segmentation" style mask for each task
    if has_saliency:
        # Truth for saliency-task
        true_saliency = np.zeros(shape, dtype=np.uint8)
        saliency_weights = np.ones(shape, dtype=np.float32)
        for true_sseg, true_catname in zip(true_ssegs, true_catnames):
            if true_catname not in background_classes:
                if true_catname in ignore_classes:
                    # background should be background
                    saliency_weights = true_sseg.fill(saliency_weights, value=0)
                elif true_catname in context_classes:
                    # Ignore context classes in saliency
                    # Ignore no-activity and post-construction, ignore, and Unknown
                    saliency_weights = true_sseg.fill(saliency_weights, value=0)
                else:
                    # Score positive, site prep, and active construction.
                    true_saliency = true_sseg.fill(true_saliency, value=1)

    if classes_of_interest:
        # Truth for class-task
        catname_to_true: Dict[str, np.ndarray] = {
            catname: np.zeros(shape, dtype=np.float32)
            for catname in classes_of_interest
        }
        class_weights = np.ones(shape, dtype=np.float32)
        for true_sseg, true_catname in zip(true_ssegs, true_catnames):
            if true_catname not in background_classes:
                if true_catname in ignore_classes:
                    class_weights = true_sseg.fill(class_weights, value=0)
                elif true_catname in undistinguished_classes:
                    class_weights = true_sseg.fill(class_weights, value=0)
                else:
                    catname_to_true[true_catname] = true_sseg.fill(catname_to_true[true_catname], value=1)

    if classes_of_interest:
        try:
            # handle multiclass case
            pred_chan_of_interest = '|'.join(classes_of_interest)
            delayed_probs = pred_coco_img.delay(pred_chan_of_interest, space=score_space)
            class_probs = delayed_probs.finalize(as_xarray=True, nodata='auto')
            invalid_mask = np.isnan(class_probs).all(axis=2)
            class_weights[invalid_mask] = 0

            catname_to_prob = {}
            cx_to_binvecs = {}
            for cx, cname in enumerate(classes_of_interest):
                is_true = catname_to_true[cname]
                score = class_probs.loc[:, :, cname].data.copy()
                invalid_mask = np.isnan(score)
                weights = class_weights.copy()
                weights[invalid_mask] = 0
                score[invalid_mask] = 0

                pred_score = score.ravel()
                if left_bin_edges is not None:
                    # round scores down to the nearest bin
                    rounded_idx = np.searchsorted(left_bin_edges, pred_score)
                    pred_score = left_bin_edges[rounded_idx]

                catname_to_prob[cname] = score
                bin_data = {
                    # is_true denotes if the true class of the item is the
                    # category of interest.
                    'is_true': is_true.ravel(),
                    'pred_score': pred_score,
                    'weight': weights.ravel(),
                }
                bin_data = kwarray.DataFrameArray(bin_data)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes_of_interest)
                # TODO: use me?
                # bin_measures = bin_cfsn.measures()
                # bin_measures.summary()
                cx_to_binvecs[cname] = bin_cfsn
            ovr_cfns = OneVsRestConfusionVectors(cx_to_binvecs, classes_of_interest)
            class_measures = ovr_cfns.measures()
            row['mAP'] = class_measures['mAP']
            row['mAUC'] = class_measures['mAUC']
            info.update({
                # TODO: data for visualization
                'class_measures': class_measures,
                'catname_to_true': catname_to_true,
                'catname_to_prob': catname_to_prob,
            })
        except Exception:
            raise

    if has_saliency:
        try:
            # TODO: consolidate this with above class-specific code
            salient_delay = pred_coco_img.delay(salient_class, space=score_space)
            salient_prob = salient_delay.finalize(nodata='auto')[..., 0]
            invalid_mask = np.isnan(salient_prob)
            salient_prob[invalid_mask] = 0
            saliency_weights[invalid_mask] = 0

            pred_score = salient_prob.ravel()
            if left_bin_edges is not None:
                rounded_idx = np.searchsorted(left_bin_edges, pred_score)
                pred_score = left_bin_edges[rounded_idx]

            bin_cfns = BinaryConfusionVectors(kwarray.DataFrameArray({
                'is_true': true_saliency.ravel(),
                'pred_score': pred_score,
                'weight': saliency_weights.ravel().astype(np.float32),
            }))
            salient_measures = bin_cfns.measures()
            salient_summary = salient_measures.summary()

            salient_metrics = {
                'salient_' + k: v
                for k, v in ub.dict_isect(salient_summary, {
                    'ap', 'auc', 'max_f1'}).items()
            }
            row.update(salient_metrics)

            info.update({
                'salient_measures': salient_measures,
                'salient_prob': salient_prob,
                'true_saliency': true_saliency,
            })

            if 1:
                maximized_info = salient_measures.maximized_thresholds()

                # This cherry-picks a threshold per image!
                cherry_picked_thresh = maximized_info['f1']['thresh']
                pred_saliency = salient_prob > cherry_picked_thresh

                y_true = true_saliency.ravel()
                y_pred = pred_saliency.ravel()
                sample_weight = saliency_weights.ravel()
                mat = skm.confusion_matrix(y_true, y_pred, labels=np.array([0, 1]),
                                           sample_weight=sample_weight)
                info.update({
                    'mat': mat,
                    'pred_saliency': pred_saliency,
                    'saliency_thresh': cherry_picked_thresh,
                })
        except Exception:
            pass

    # TODO: look at the category ranking at each pixel by score.
    # Is there a generalization of a confusion matrix to a ranking tensor?
    # if 0:
    #     # TODO: Reintroduce hard-polygon segmentation scoring?
    #     # Score hard-threshold predicted annotations
    #     # SCORE PREDICTED ANNOTATIONS
    #     # Create a pred "panoptic segmentation" style mask
    #     pred_saliency = np.zeros(shape, dtype=np.uint8)
    #     pred_dets = pred_coco.annots(gid=gid2).detections
    #     for pred_sseg in pred_dets.data['segmentations']:
    #         pred_saliency = pred_sseg.fill(pred_saliency, value=1)
    return info


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


@profile
def colorize_class_probs(probs, classes):
    """
    probs = pred_cat_ohe
    classes = pred_classes
    """
    # color = classes.graph.nodes[node].get('color', None)

    # Define default colors
    # default_cidx_to_color = kwimage.Color.distinct(len(data))

    # try and read colors from classes CategoryTree
    # try:
    #     cidx_to_color = []

    cidx_to_color = []
    for cidx in range(len(probs)):
        node = classes[cidx]
        color = classes.graph.nodes[node].get('color', None)
        if color is not None:
            color = kwimage.Color(color).as01()
        cidx_to_color.append(color)

    import distinctipy
    have_colors = [c for c in cidx_to_color if c is not None]
    num_need = sum(c is None for c in cidx_to_color)
    if num_need:
        new_colors = distinctipy.get_colors(
            num_need, exclude_colors=have_colors, rng=569944)
        new_color_iter = iter(new_colors)
        cidx_to_color = [next(new_color_iter) if c is None else c for c in cidx_to_color]

    canvas_dtype = np.float32

    # Each class gets its own color, and modulates the alpha
    h, w = probs.shape[-2:]
    layer_shape = (h, w, 4)
    background = np.zeros(layer_shape, dtype=canvas_dtype)
    background[..., 3] = 1.0
    layers = []
    for cidx, chan in enumerate(probs):
        color = cidx_to_color[cidx]
        layer = np.empty(layer_shape, dtype=canvas_dtype)
        layer[..., 3] = chan
        layer[..., 0:3] = color
        layers.append(layer)
    layers.append(background)

    colormask = kwimage.overlay_alpha_layers(
        layers, keepalpha=False, dtype=canvas_dtype)

    return colormask


@profile
def draw_truth_borders(true_dets, canvas, alpha=1.0, color=None):
    true_sseg = true_dets.data['segmentations']
    true_cidxs = true_dets.data['class_idxs']
    _classes = true_dets.data['classes']

    if color is None:
        _nodes = ub.take(_classes.idx_to_node, true_cidxs)
        _node_data = ub.take(_classes.graph.nodes, _nodes)
        _node_colors = [d['color'] for d in _node_data]
        color = _node_colors

    canvas = kwimage.ensure_float01(canvas)
    if alpha < 1.0:
        # remove this condition when kwimage 0.8.3 is released always take else
        empty_canvas = np.zeros_like(canvas, shape=(canvas.shape[0:2] + (4,)))
        overlay_canvas = true_sseg.draw_on(empty_canvas, fill=False,
                                           border=True, color=color, alpha=1.0)
        overlay_canvas[..., 3] *= alpha
        canvas = kwimage.overlay_alpha_images(overlay_canvas, canvas)
    else:
        canvas = true_sseg.draw_on(canvas, fill=False, border=True,
                                   color=color, alpha=alpha)
    return canvas


@profile
def dump_chunked_confusion(full_classes, true_coco_imgs, chunk_info,
                           heatmap_dpath, score_space='video', title=None):
    """
    Draw a a sequence of true/pred image predictions
    """
    from watch import heuristics
    color_labels = ['TN', 'TP', 'FN', 'FP']
    colors = list(ub.take(heuristics.CONFUSION_COLOR_SCHEME, color_labels))
    # colors = ['blue', 'green', 'yellow', 'red']
    # colors = ['black', 'white', 'yellow', 'red']
    color_lut = np.array([kwimage.Color(c).as255() for c in colors])
    # full_classes: kwcoco.CategoryTree = true_coco.object_categories()

    # Make a legend
    color01_lut = color_lut / 255.0
    legend_images = []

    if 'catname_to_prob' in chunk_info[0]:
        # Class Legend
        label_to_color = {
            node: kwimage.Color(data['color']).as01()
            for node, data in full_classes.graph.nodes.items()}
        label_to_color = ub.sorted_keys(label_to_color)
        legend_img_class = _memo_legend(label_to_color)
        legend_images.append(legend_img_class)

    if 'pred_saliency' in chunk_info[0]:
        # Confusion Legend
        label_to_color = ub.dzip(color_labels, color01_lut)
        legend_img_saliency_cfsn = _memo_legend(label_to_color)
        legend_img_saliency_cfsn = kwimage.ensure_uint255(legend_img_saliency_cfsn)
        legend_images.append(legend_img_saliency_cfsn)

    if len(legend_images):
        legend_img = kwimage.stack_images(legend_images, axis=0, pad=5)
    else:
        legend_img = None

    # Draw predictions on each frame
    parts = []
    frame_nums = []
    true_gids = []
    unique_vidnames = set()
    for info, true_coco_img in zip(chunk_info, true_coco_imgs):
        row = info['row']
        if row.get('video', ''):
            unique_vidnames.add(row['video'])

        # true_gid = row['true_gid']
        # true_coco_img = true_coco.coco_image(true_gid)
        true_gid = true_coco_img.img['id']

        true_img = true_coco_img.img
        frame_index = true_img.get('frame_index', None)
        if frame_index is not None:
            frame_nums.append(frame_index)
        true_gids.append(true_gid)

        # image_header_text = f'{frame_index} - gid = {true_gid}'

        header_lines = heuristics.build_image_header_text(
            img=true_img,
            name=None,
            _header_extra=None,
        )
        # date_captured = true_img.get('date_captured', '')
        # frame_index = true_img.get('frame_index', None)
        # gid = true_img.get('id', None)
        # sensor_coarse = true_img.get('sensor_coarse', 'unknown')
        # _header_extra = None
        # header_line_infos = [
        #     [f'gid={gid}, frame={frame_index}', _header_extra],
        #     [sensor_coarse, date_captured],
        # ]
        # header_lines = []
        # for line_info in header_line_infos:
        #     header_line = ' '.join([p for p in line_info if p])
        #     if header_line:
        #         header_lines.append(header_line)
        image_header_text = '\n'.join(header_lines)

        imgw = info['shape'][1]
        # SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch=52-step=2269088
        header = kwimage.draw_header_text(
            {'width': imgw},
            # image=confusion_image,
            # image=None,
            text=image_header_text, color='red', stack=False)

        vert_parts = [
            header,
        ]

        if 'catname_to_prob' in info:

            true_dets = info['true_dets']
            true_dets.data['classes'] = full_classes

            # from watch import heuristics
            pred_classes = kwcoco.CategoryTree.coerce(list(info['catname_to_prob'].keys()))
            true_classes = kwcoco.CategoryTree.coerce(list(info['catname_to_true'].keys()))
            # todo: ensure colors are robust and consistent
            for node in pred_classes.graph.nodes():
                pred_classes.graph.nodes[node]['color'] = full_classes.graph.nodes[node]['color']
            for node in true_classes.graph.nodes():
                true_classes.graph.nodes[node]['color'] = full_classes.graph.nodes[node]['color']

            # pred_classes = kwcoco.CategoryTree
            pred_cat_ohe = np.stack(list(info['catname_to_prob'].values()))
            true_cat_ohe = np.stack(list(info['catname_to_true'].values()))
            # class_pred_idx = pred_cat_ohe.argmax(axis=0)
            # class_true_idx = true_cat_ohe.argmax(axis=0)

            true_overlay = colorize_class_probs(true_cat_ohe, true_classes)[..., 0:3]
            # true_heatmap = kwimage.Heatmap(class_probs=true_cat_ohe, classes=true_classes)
            # true_overlay = true_heatmap.colorize('class_probs')[..., 0:3]
            true_overlay = draw_truth_borders(true_dets, true_overlay, alpha=1.0)
            true_overlay = kwimage.ensure_uint255(true_overlay)
            true_overlay = kwimage.draw_text_on_image(
                true_overlay, 'true class', org=(1, 1), valign='top',
                color='limegreen', border=1)
            vert_parts.append(true_overlay)

            pred_overlay = colorize_class_probs(pred_cat_ohe, pred_classes)[..., 0:3]
            # pred_heatmap = kwimage.Heatmap(class_probs=pred_cat_ohe, classes=pred_classes)
            # pred_overlay = pred_heatmap.colorize('class_probs')[..., 0:3]
            pred_overlay = draw_truth_borders(true_dets, pred_overlay, alpha=0.05, color='white')
            # pred_overlay = draw_truth_borders(true_dets, pred_overlay, alpha=0.05)
            pred_overlay = kwimage.ensure_uint255(pred_overlay)
            pred_overlay = kwimage.draw_text_on_image(
                pred_overlay, 'pred class', org=(1, 1), valign='top',
                color='dodgerblue', border=1)
            vert_parts.append(pred_overlay)

        if 'pred_saliency' in info:
            pred_saliency = info['pred_saliency'].astype(np.uint8)
            true_saliency = info['true_saliency']
            saliency_thresh = info['saliency_thresh']
            confusion_idxs = utils.confusion_image(pred_saliency, true_saliency)
            confusion_image = color_lut[confusion_idxs]
            confusion_image = kwimage.ensure_uint255(confusion_image)
            confusion_image = kwimage.draw_text_on_image(
                confusion_image,
                f'confusion saliency: thresh={saliency_thresh:0.3f}',
                org=(1, 1), valign='top',
                color='white', border=1)
            vert_parts.append(
                confusion_image
            )
        elif 'true_saliency' in info:
            true_saliency = info['true_saliency']
            true_saliency = true_saliency.astype(np.float32)
            heatmap = kwimage.make_heatmask(
                true_saliency, with_alpha=0.5, cmap='plasma')
            # heatmap[invalid_mask] = 0
            heatmap_int = kwimage.ensure_uint255(heatmap[..., 0:3])
            heatmap_int = kwimage.draw_text_on_image(
                heatmap_int, 'true saliency', org=(1, 1), valign='top',
                color='limegreen', border=1)
            vert_parts.append(heatmap_int)
        # confusion_image = kwimage.draw_text_on_image(
        #     confusion_image, image_text, org=(1, 1), valign='top',
        #     color='white', border={'color': 'black'})

        # TODO:
        # Can we show the reference image?
        # TODO:
        # Show the datetime on the top of the image (and the display band?)
        real_image_norm = None
        real_image_int = None

        TRY_IMREAD = 1
        if TRY_IMREAD:
            avali_chans = {p2 for p1 in true_coco_img.channels.spec.split(',') for p2 in p1.split('|')}
            chosen_viz_channs = None
            if len(avali_chans & {'red', 'green', 'blue'}) == 3:
                chosen_viz_channs = 'red|green|blue'
            elif len(avali_chans & {'r', 'g', 'b'}) == 3:
                chosen_viz_channs = 'r|g|b'
            else:
                chosen_viz_channs = true_coco_img.primary_asset()['channels']
            try:
                real_image = true_coco_img.delay(chosen_viz_channs, space=score_space).finalize(nodata='auto')
                real_image_norm = kwimage.normalize_intensity(real_image)
                real_image_int = kwimage.ensure_uint255(real_image_norm)
            except Exception as ex:
                print('ex = {!r}'.format(ex))

        TRY_SOFT = 1
        salient_prob = None
        if TRY_SOFT:
            salient_prob = info.get('salient_prob', None)
            # invalid_mask = info.get('invalid_mask', None)
            if salient_prob is not None:
                heatmap = kwimage.make_heatmask(
                    salient_prob, with_alpha=0.5, cmap='plasma')
                # heatmap[invalid_mask] = 0
                heatmap_int = kwimage.ensure_uint255(heatmap[..., 0:3])
                heatmap_int = kwimage.draw_text_on_image(
                    heatmap_int, 'pred saliency', org=(1, 1), valign='top',
                    color='dodgerblue', border=1)
                vert_parts.append(heatmap_int)
                # if real_image_norm is not None:
                #     overlaid = kwimage.overlay_alpha_layers([heatmap, real_image_norm.mean(axis=2)])
                #     overlaid = kwimage.ensure_uint255(overlaid[..., 0:3])
                #     vert_parts.append(overlaid)

        if real_image_int is not None:
            vert_parts.append(real_image_int)

        vert_parts = [kwimage.ensure_uint255(c) for c in vert_parts]
        vert_stack = kwimage.stack_images(vert_parts, axis=0)
        parts.append(vert_stack)

    max_frame = None if len(frame_nums) == 0 else max(frame_nums)
    min_frame = None if len(frame_nums) == 0 else min(frame_nums)
    max_gid = max(true_gids)
    min_gid = min(true_gids)

    if max_frame == min_frame:
        frame_part = f'{min_frame}'
    else:
        frame_part = f'{min_frame}-{max_frame}'

    if max_gid == min_gid:
        gid_part = f'{min_gid}'
    else:
        gid_part = f'{min_gid}-{max_gid}'
    vidname_part = '_'.join(list(unique_vidnames))
    if not vidname_part:
        vidname_part = '_loose_images'

    plot_fstem = f'{vidname_part}-{frame_part}-{gid_part}'

    canvas_title_parts = []
    if title:
        canvas_title_parts.append(title)
    canvas_title_parts.append(plot_fstem)
    canvas_title = '\n'.join(canvas_title_parts)

    plot_canvas = kwimage.stack_images(parts, axis=1, overlap=-10)

    if legend_img is not None:
        plot_canvas = kwimage.stack_images(
            [plot_canvas, legend_img], axis=1, overlap=-10)

    header = kwimage.draw_header_text(
        {'width': plot_canvas.shape[1]}, canvas_title)
    plot_canvas = kwimage.stack_images([header, plot_canvas], axis=0)

    heatmap_dpath = ub.Path(str(heatmap_dpath))
    vid_plot_dpath = (heatmap_dpath / vidname_part).ensuredir()
    plot_fpath = vid_plot_dpath / (plot_fstem + '.jpg')
    kwimage.imwrite(str(plot_fpath), plot_canvas)


@profile
def evaluate_segmentations(true_coco, pred_coco, eval_dpath=None,
                           draw_curves='auto', draw_heatmaps='auto',
                           score_space='video', workers='auto',
                           draw_workers='auto'):
    """
    CommandLine:
        XDEV_PROFILE=1 xdoctest -m watch.tasks.fusion.evaluate evaluate_segmentations

    Example:
        >>> from watch.tasks.fusion.evaluate import *  # NOQA
        >>> from kwcoco.coco_evaluator import CocoEvaluator
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> import kwcoco
        >>> true_coco1 = kwcoco.CocoDataset.demo('vidshapes2')
        >>> true_coco2 = kwcoco.CocoDataset.demo('shapes8')
        >>> #true_coco1 = kwcoco.CocoDataset.demo('vidshapes9')
        >>> #true_coco2 = kwcoco.CocoDataset.demo('shapes128')
        >>> true_coco = kwcoco.CocoDataset.union(true_coco1, true_coco2)
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>>     'with_probs': True,
        >>>     'with_heatmaps': True,
        >>>     'verbose': 1,
        >>> }
        >>> # TODO: it would be nice to demo the soft metrics
        >>> # functionality by adding "salient_prob" or "class_prob"
        >>> # auxiliary channels to this demodata.
        >>> print('perterbing')
        >>> pred_coco = perterb_coco(true_coco, **kwargs)
        >>> eval_dpath = ub.ensure_app_cache_dir('watch/tests/fusion_eval')
        >>> print('eval_dpath = {!r}'.format(eval_dpath))
        >>> score_space = 'image'
        >>> draw_curves = 'auto'
        >>> draw_heatmaps = 'auto'
        >>> #draw_heatmaps = False
        >>> workers = 'min(avail-2,6)'
        >>> #workers = 0
        >>> evaluate_segmentations(true_coco, pred_coco, eval_dpath,
        >>>                        score_space=score_space,
        >>>                        draw_heatmaps=draw_heatmaps,
        >>>                        draw_curves=draw_curves, workers=workers)
    """
    import platform
    from watch import heuristics
    from watch.utils.lightning_ext import util_globals

    # Ensure each class has colors.
    heuristics.ensure_heuristic_coco_colors(true_coco)
    true_classes = list(true_coco.object_categories())
    full_classes: kwcoco.CategoryTree = true_coco.object_categories()

    # Sometimes supercategories dont get colors, this fixes that.
    heuristics.ensure_heuristic_category_tree_colors(full_classes)

    workers = util_globals.coerce_num_workers(workers)
    if draw_workers == 'auto':
        draw_workers = min(2, workers)
    else:
        draw_workers = util_globals.coerce_num_workers(draw_workers)

    # Extract metadata about the predictions to persist
    meta = {}
    meta['info'] = info = []

    # Add info about where and when evaluation happened
    meta['hostname'] = platform.node()
    meta['user'] = ub.Path(ub.userhome()).name
    meta['time'] = ub.timestamp()

    if pred_coco.fpath is not None:
        pred_fpath = ub.Path(pred_coco.fpath)
        meta['pred_name'] = '_'.join((list(pred_fpath.parts[-2:-1]) + [pred_fpath.stem]))

    predicted_info = pred_coco.dataset.get('info', [])
    for item in predicted_info:
        if item.get('type', None) == 'measure':
            info.append(item)
        if item.get('type', None) == 'process':
            proc_name = item.get('properties', {}).get('name', None)
            if proc_name == 'watch.tasks.fusion.predict':
                package_fpath = item['properties']['args'].get('package_fpath')
                if 'title' not in item:
                    item['title'] = ub.Path(package_fpath).stem
                if 'package_name' not in item:
                    item['package_name'] = ub.Path(package_fpath).stem

                # FIXME: title should also include pred-config info

                meta['title'] = item['title']
                meta['package_name'] = item['package_name']
                info.append(item)

    # Title contains the model package name if we can infer it
    package_name = meta.get('package_name', '')
    pred_name = meta.get('pred_name', '')
    title_parts = [p for p in [package_name, pred_name] if p]
    meta['title_parts'] = title_parts
    title = meta['title'] = ' - '.join(title_parts)

    required_marked = 'auto'  # parametarize
    if required_marked == 'auto':
        # In "auto" mode dont require marks if all images are unmarked,
        # otherwise assume that we should restirct to marked images
        required_marked = any(pred_coco.images().lookup('has_predictions', False))

    matches  = kwcoco_extensions.associate_images(
        true_coco, pred_coco, key_fallback='id')

    video_matches = matches['video']
    image_matches = matches['image']

    n_vid_matches = len(video_matches)
    n_img_per_vid_matches = [len(d['match_gids1']) for d in video_matches]
    n_img_matches = image_matches['match_gids1']
    print('n_img_per_vid_matches = {}'.format(ub.repr2(n_img_per_vid_matches, nl=1)))
    print('n_vid_matches = {}'.format(ub.repr2(n_vid_matches, nl=1)))
    print('n_img_matches = {!r}'.format(n_img_matches))

    rows = []
    chunk_size = 5
    # thresh_bins = 256 * 256
    # thresh_bins = 64 * 64
    thresh_bins = np.linspace(0, 1, 128 * 128)  # this is more stable using an ndarray

    if draw_curves == 'auto':
        draw_curves = bool(eval_dpath is not None)

    if draw_heatmaps == 'auto':
        draw_heatmaps = bool(eval_dpath is not None)

    if eval_dpath is None:
        heatmap_dpath = None
    else:
        heatmap_dpath = ub.Path(eval_dpath) / 'heatmaps'
        heatmap_dpath.mkdir(exist_ok=True, parents=True)

    # Objects that will aggregate confusion across multiple images
    salient_measure_combiner = MeasureCombiner(thresh_bins=thresh_bins)
    class_measure_combiner = OneVersusRestMeasureCombiner(thresh_bins=thresh_bins)

    # Gather the true and predicted image pairs to be scored
    total_images = 0
    if required_marked:
        for video_match in video_matches:
            gids1 = video_match['match_gids1']
            gids2 = video_match['match_gids2']
            flags = pred_coco.images(gids2).lookup('has_predictions', False)
            video_match['match_gids1'] = list(ub.compress(gids1, flags))
            video_match['match_gids2'] = list(ub.compress(gids2, flags))
            total_images += len(gids1)
        gids1 = image_matches['match_gids1']
        gids2 = image_matches['match_gids2']
        flags = pred_coco.images(gids2).lookup('has_predictions', False)
        image_matches['match_gids1'] = list(ub.compress(gids1, flags))
        image_matches['match_gids2'] = list(ub.compress(gids2, flags))
        total_images += len(gids1)
    else:
        total_images = None

    # Prepare job pools
    print('workers = {!r}'.format(workers))
    print('draw_workers = {!r}'.format(draw_workers))
    workers = 10
    metrics_executor = ub.Executor(mode='process', max_workers=workers)
    draw_executor = ub.Executor(mode='process', max_workers=draw_workers)

    prog = ub.ProgIter(total=total_images, desc='submit scoring jobs', adjust=False, freq=1)
    prog.begin()

    job_chunks = []
    draw_jobs = []

    # Submit scoring jobs over pairs of true-predicted images in videos
    for video_match in video_matches:
        prog.set_extra('comparing ' + video_match['vidname'])
        gids1 = video_match['match_gids1']
        gids2 = video_match['match_gids2']
        if required_marked:
            flags = pred_coco.images(gids2).lookup('has_predictions', False)
            gids1 = list(ub.compress(gids1, flags))
            gids2 = list(ub.compress(gids2, flags))

        current_chunk = []
        for gid1, gid2 in zip(gids1, gids2):
            pred_coco_img = pred_coco.coco_image(gid1).detach()
            true_coco_img = true_coco.coco_image(gid2).detach()
            true_dets = true_coco.annots(gid=gid1).detections

            vidid1 = true_coco.imgs[gid1]['video_id']
            video1 = true_coco.index.videos[vidid1]

            job = metrics_executor.submit(
                single_image_segmentation_metrics, pred_coco_img,
                true_coco_img, true_classes, true_dets, video1,
                score_space=score_space, thresh_bins=thresh_bins)

            if len(current_chunk) >= chunk_size:
                job_chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append(job)
            prog.update()

        if len(current_chunk) > 0:
            job_chunks.append(current_chunk)

    # Submit scoring jobs over pairs of true-predicted images without videos
    if score_space == 'image':
        gids1 = image_matches['match_gids1']
        gids2 = image_matches['match_gids2']

        for gid1, gid2 in zip(gids1, gids2):
            pred_coco_img = pred_coco.coco_image(gid1).detach()
            true_coco_img = true_coco.coco_image(gid2).detach()
            true_dets = true_coco.annots(gid=gid1).detections
            video1 = None
            job = metrics_executor.submit(
                single_image_segmentation_metrics, pred_coco_img,
                true_coco_img, true_classes, true_dets, video1,
                score_space=score_space, thresh_bins=thresh_bins)
            prog.update()
            job_chunks.append([job])
    else:
        if len(image_matches['match_gids1']) > 0:
            warnings.warn(ub.paragraph(
                '''
                Scoring was requested in video mode, but there are
                {len(image_matches['match_gids1'])} true/pred image pairs that
                are unassociated with a video. These pairs will not be included
                in video space scoring.
                '''))
    prog.end()

    num_jobs = sum(map(len, job_chunks))

    RICH_PROG = 'auto'
    if RICH_PROG == 'auto':
        # Use rich outside of slurm
        RICH_PROG = not os.environ.get('SLURM_JOBID', '')

    if RICH_PROG:
        import rich
        import rich.progress
        progress = rich.progress.Progress(
            "[progress.description]{task.description}",
            rich.progress.BarColumn(),
            rich.progress.MofNCompleteColumn(),
            # "[progress.percentage]{task.percentage:>3.0f}%",
            rich.progress.TimeRemainingColumn(),
            rich.progress.TimeElapsedColumn(),
        )
    else:
        score_prog = ub.ProgIter(desc='Scoring',  total=num_jobs)
        num_draw_finished = 1
        progress = score_prog  # Hack

    DEBUG = 0
    if DEBUG:
        orig_infos = []

    with progress:
        if RICH_PROG:
            score_task = progress.add_task("[cyan] Scoring...", total=num_jobs)
            if draw_heatmaps:
                draw_task = progress.add_task("[green] Drawing...", total=len(job_chunks))

        for job_chunk in job_chunks:
            chunk_info = []
            for job in job_chunk:
                info = job.result()
                if DEBUG:
                    orig_infos.append(info)

                if RICH_PROG:
                    progress.update(score_task, advance=1)
                else:
                    score_prog.update(1)
                rows.append(info['row'])

                class_measures = info.get('class_measures', None)
                salient_measures = info.get('salient_measures', None)
                if salient_measures is not None:
                    salient_measure_combiner.submit(salient_measures)
                if class_measures is not None:
                    class_measure_combiner.submit(class_measures)
                if draw_heatmaps:
                    chunk_info.append(info)

            # Once a job chunk is done, clear its memory
            job = None
            job_chunk.clear()

            # Reduce measures over the chunk
            if salient_measure_combiner.queue_size > chunk_size:
                salient_measure_combiner.combine()
            if class_measure_combiner.queue_size > chunk_size:
                class_measure_combiner.combine()

            if draw_heatmaps:
                # Let the draw executor release any memory it can
                remaining_draw_jobs = []
                for draw_job in draw_jobs:
                    if draw_job.done():
                        draw_job.result()
                        if RICH_PROG:
                            progress.update(draw_task, advance=1)
                        else:
                            num_draw_finished += 1
                            score_prog.set_extra(f'Drawing : {num_draw_finished}')
                            pass
                    else:
                        remaining_draw_jobs.append(draw_job)
                draw_job = None
                draw_jobs = remaining_draw_jobs

                # As chunks of evaluation jobs complete, submit background jobs to
                # draw results to disk if requested.
                true_gids = [info['row']['true_gid'] for info in chunk_info]
                true_coco_imgs = true_coco.images(true_gids).coco_images
                true_coco_imgs = [g.detach() for g in true_coco_imgs]
                draw_job = draw_executor.submit(
                    dump_chunked_confusion, full_classes, true_coco_imgs,
                    chunk_info, heatmap_dpath, score_space=score_space,
                    title=title)
                draw_jobs.append(draw_job)

        metrics_executor.shutdown()

        if draw_heatmaps:
            # Allow all drawing jobs to finalize
            while draw_jobs:
                job = draw_jobs.pop()
                job.result()
                if RICH_PROG:
                    progress.update(draw_task, advance=1)
                else:
                    # draw_prog.update()
                    num_draw_finished += 1
                    score_prog.set_extra(f'Drawing : {num_draw_finished}')
            draw_executor.shutdown()

    df = pd.DataFrame(rows)
    print('Per Image Pixel Measures')
    print(df)
    print(df.describe().T)

    # Finalize all of the aggregated measures
    print('Finalize salient measures')
    # Note: this will return False if there are no salient measures
    salient_combo_measures = salient_measure_combiner.finalize()
    if salient_combo_measures is False or salient_combo_measures is None:
        # Use nan measures from empty binary confusion vectors
        salient_combo_measures = BinaryConfusionVectors(None).measures()
    # print('salient_combo_measures = {!r}'.format(salient_combo_measures))

    if DEBUG:
        # Redo salient combine
        tocombine = []

        for p in tocombine:
            z = ub.dict_isect(p, {'fp_count', 'tp_count', 'fn_count', 'tn_count', 'thresholds', 'nsupport'})
            print(ub.repr2(ub.map_vals(list, z), nl=0))

        salient_measure_combiner = MeasureCombiner(thresh_bins=thresh_bins)
        print('salient_combo_measures.__dict__ = {!r}'.format(salient_combo_measures.__dict__))
        # precision = None
        # growth = None
        from kwcoco.metrics.confusion_measures import Measures
        for info in orig_infos:
            class_measures = info.get('class_measures', None)
            salient_measures = info.get('salient_measures', None)
            if salient_measures is not None:
                tocombine.append(salient_measures)
                salient_measure_combiner.submit(salient_measures)

        combo = Measures.combine(tocombine, thresh_bins=thresh_bins).reconstruct()
        print('combo = {!r}'.format(combo))

        combo = Measures.combine(tocombine, precision=2)
        combo.reconstruct()
        print('combo = {!r}'.format(combo))

        combo = Measures.combine(tocombine, growth='max')
        combo.reconstruct()
        print('combo = {!r}'.format(combo))

        salient_combo_measures = salient_measure_combiner.finalize()
        print('salient_combo_measures = {!r}'.format(salient_combo_measures))

    print('Finalize class measures')
    class_combo_measure_dict = class_measure_combiner.finalize()
    ovr_combo_measures = class_combo_measure_dict['perclass']

    # Combine class + salient measures using the "SingleResult" container
    # (TODO: better API)
    result = CocoSingleResult(
        salient_combo_measures, ovr_combo_measures, None, meta)
    print('result = {}'.format(result))

    if salient_combo_measures is not None:
        if eval_dpath is not None:
            curve_dpath = ub.Path(eval_dpath) / 'curves'
            curve_dpath.mkdir(exist_ok=True, parents=True)

            if isinstance(salient_combo_measures, dict):
                salient_combo_measures['meta'] = meta

            title = '\n'.join(meta.get('title_parts', [meta.get('title', '')]))
            measures_fpath2 = curve_dpath / 'measures2.json'
            print('Dump measures_fpath2={}'.format(measures_fpath2))
            result.dump(os.fspath(measures_fpath2))

            if draw_curves:
                import kwplot
                # kwplot.autompl()
                with kwplot.BackendContext('agg'):
                    fig = kwplot.figure(doclf=True)

                    print('Dump salient figures')
                    salient_combo_measures.summary_plot(fnum=1, title=title)
                    fig = kwplot.autoplt().gcf()
                    fig.savefig(str(curve_dpath / 'salient_summary.png'))

                    print('Dump class figures')
                    result.dump_figures(curve_dpath, expt_title=title)

    summary = {}
    if class_combo_measure_dict is not None:
        summary['class_mAP'] = class_combo_measure_dict['mAP']
        summary['class_mAUC'] = class_combo_measure_dict['mAUC']

    if salient_combo_measures is not None:
        summary['salient_ap'] = salient_combo_measures['ap']
        summary['salient_auc'] = salient_combo_measures['auc']
        summary['salient_max_f1'] = salient_combo_measures['max_f1']

    print('summary = {}'.format(ub.repr2(
        summary, nl=1, precision=4, align=':', sort=0)))
    print('eval_dpath = {!r}'.format(eval_dpath))
    return df


def _redraw_measures(eval_dpath):
    """
    hack helper for developer, not critical
    """
    curve_dpath = ub.Path(eval_dpath) / 'curves'
    measures_fpath = curve_dpath / 'measures.json'
    with open(measures_fpath, 'r') as file:
        state = json.load(file)
        salient_combo_measures = Measures.from_json(state)
        meta = salient_combo_measures.get('meta', [])
        title = ''
        if meta is not None:
            if isinstance(meta, list):
                # Old
                for item in meta:
                    title = item.get('title', title)
            else:
                # title = meta.get('title', title)
                title = '\n'.join(meta.get('title_parts', [meta.get('title', '')]))
        import kwplot
        with kwplot.BackendContext('agg'):
            salient_combo_measures.summary_plot(fnum=1, title=title)
            fig = kwplot.autoplt().gcf()
            fig.savefig(str(curve_dpath / 'summary_redo.png'))


if __name__ == '__main__':
    r"""
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

    python -m watch.tasks.fusion.predict \
        --write_probs=True \
        --write_preds=False \
        --with_class=auto \
        --with_saliency=auto \
        --with_change=False \
        --package_fpath=$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt \
        --pred_dataset=$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/pred_BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819/combo_vali_nowv.kwcoco/pred.kwcoco.json \
        --test_dataset=$DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json \
        --num_workers=5 \
        --compress=DEFLATE \
        --gpus=0, \
        --batch_size=8

    python -m watch.tasks.fusion.evaluate \
        --true_dataset=$DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json \
        --pred_dataset=$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/pred_BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819/combo_vali_nowv.kwcoco/pred.kwcoco.json \
          --eval_dpath=$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/pred_BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819/combo_vali_nowv.kwcoco/eval \
          --score_space=video \
          --draw_curves=1 \
          --draw_heatmaps=1


    python -m watch.tasks.fusion.predict \
        --write_probs=True \
        --write_preds=False \
        --with_class=auto \
        --with_saliency=auto \
        --with_change=False \
        --package_fpath=/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/packages/BOTH_TA1_COMBO_TINY_p1_v0100/BOTH_TA1_COMBO_TINY_p1_v0100_epoch=4-step=5119-v2.pt \
        --pred_dataset=/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/pred/BOTH_TA1_COMBO_TINY_p1_v0100/pred_BOTH_TA1_COMBO_TINY_p1_v0100_epoch=4-step=5119-v2/Drop2-Aligned-TA1-2022-02-15_combo_DILM_nowv_vali.kwcoco/unknown_pred_cfg/pred.kwcoco.json \
        --test_dataset=/home/joncrall/data/dvc-repos/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/combo_DILM_nowv_vali.kwcoco.json \
        --num_workers=5 \
        --compress=DEFLATE \
        --gpus=0, \
        --batch_size=1

    python -m watch.tasks.fusion.evaluate \
        --true_dataset=/home/joncrall/data/dvc-repos/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/combo_DILM_nowv_vali.kwcoco.json \
        --pred_dataset=/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/pred/BOTH_TA1_COMBO_TINY_p1_v0100/pred_BOTH_TA1_COMBO_TINY_p1_v0100_epoch=4-step=5119-v2/Drop2-Aligned-TA1-2022-02-15_combo_DILM_nowv_vali.kwcoco/unknown_pred_cfg/pred.kwcoco.json \
          --eval_dpath=/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/eval/BOTH_TA1_COMBO_TINY_p1_v0100/pred_BOTH_TA1_COMBO_TINY_p1_v0100_epoch=4-step=5119-v2/Drop2-Aligned-TA1-2022-02-15_combo_DILM_nowv_vali.kwcoco/unknown_pred_cfg/eval \
          --score_space=video \
          --draw_curves=1 \
          --draw_heatmaps=1 --workers=2


    PRED_DATASET=$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/pred/Drop3_BASELINE_BAS_V303/pred_Drop3_BASELINE_BAS_V303_epoch=5-step=12287/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json
    TEST_DATASET=$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json
    python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET \
        --eval_dpath=tmp/eval-tmp \
        --score_space=video \
        --draw_curves=1 \
        --draw_heatmaps=1 --workers=2 --dump=./debug-eval.yaml

    Ignore:
        kwargs = {
            'config': './debug-eval.yaml',
        }
        cmdline = False

    V303_epoch=5-step=12287


    """
    # import xdev
    # xdev.make_warnings_print_tracebacks()
    main()
