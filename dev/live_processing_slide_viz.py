import kwimage
import numpy as np  # NOQA
import kwcoco
from watch import heuristics
import kwplot
kwplot.autompl()

shape = (128, 128)
# init_canvas0 = np.zeros(shape, dtype=np.float32)
# poly1 = kwimage.Polygon.random().scale(512, 512)

classes = kwcoco.CategoryTree.coerce(['Active Construction', 'Site Preparation', 'background'])
heuristics.ensure_heuristic_category_tree_colors(classes)


init_heatmap0 = kwimage.Heatmap.random(shape, classes=classes, noise=0.1, smooth_k=7, ensure_background=False)

init_canvas0 = init_heatmap0.draw_on(imgspace=False, channel='class_probs', with_alpha=0.8)
init_canvas0 = init_canvas0.clip(0, 1)

new_frame0 = kwimage.Heatmap.random(shape, classes=classes, noise=0.1, smooth_k=7, ensure_background=False)
new_frame1 = kwimage.Heatmap.random(shape, classes=classes, noise=0.1, smooth_k=7, ensure_background=False)

new_canvas0 = new_frame0.draw_on(imgspace=False, channel='class_probs', with_alpha=0.8).clip(0, 1)
new_canvas1 = new_frame1.draw_on(imgspace=False, channel='class_probs', with_alpha=0.8).clip(0, 1)

final_canvas0 = (new_canvas0 + init_canvas0) / 2
final_canvas1 = new_canvas1

kwplot.imshow(init_canvas0, fnum=1, pnum=(3, 2, 1), title='existing heatmap (frame0)')

kwplot.imshow(new_canvas0, fnum=1, title='new heatmap (frame0)', pnum=(3, 2, 3))
kwplot.imshow(new_canvas1, fnum=1, title='new heatmap (frame1)', pnum=(3, 2, 4))

kwplot.imshow(final_canvas0, fnum=1, title='final heatmap (frame0)', pnum=(3, 2, 5))
kwplot.imshow(new_canvas1, fnum=1, title='final heatmap (frame1)', pnum=(3, 2, 6))


#######

# Ensemble Viz

from kwcoco.demo import toydata_video
dims = h, w = 224, 224
dset = toydata_video.random_single_video_dset((h, w), num_frames=5, max_speed=0, num_tracks=3)
dset.rename_categories({'star': 'Active Construction', 'superstar': 'Site Preparation', 'eff': 'Post Construction'})
dset.ensure_category('background')
heuristics.ensure_heuristic_coco_colors(dset)

coco_images = dset.images().coco_images

model1_preds = []
model2_preds = []


def rasterize_dets2(dets, dims):
    h, w = dims
    dets.data['score'] = np.random.rand(len(dets))
    scores = np.random.rand(len(dets))
    sseg_list = dets.data['segmentations']
    class_idxs = dets.data['class_idxs']
    background = kwimage.ensure_alpha_channel(np.zeros((h, w, 3), dtype=np.float32), alpha=1)
    stack1 = [background]
    for sseg, score, cidx in zip(sseg_list, scores, class_idxs):
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        catname = dets.classes.idx_to_node[cidx]
        color = dets.classes.graph.nodes[catname]['color']
        canvas = sseg.draw_on(canvas, color=color)
        canvas = kwimage.gaussian_blur(canvas, kernel=31)
        alpha = canvas.max(axis=2) * score
        canvas = kwimage.ensure_alpha_channel(canvas, alpha)
        canvas = kwimage.gaussian_blur(canvas, kernel=31)
        stack1.append(canvas)
    frame1 = kwimage.overlay_alpha_layers(stack1[::-1])[..., 0:3]
    frame1 = kwimage.gaussian_blur(frame1, kernel=11)
    return frame1

for img in coco_images:
    dets = dset.annots(gid=img.img['id']).detections
    dets.data.pop('keypoints', None)

    idxs1 = list(range(0, len(dets), 2))
    idxs2 = [0] + list(range(1, len(dets), 2))

    # dets1 = dets.take(idxs1)
    # dets2 = dets.take(idxs2)
    # print(f'dets1.boxes={dets1.boxes}')
    # print(f'dets2.boxes={dets2.boxes}')

    dets1 = dets2 = dets
    img_canvas1 = rasterize_dets2(dets1, dims)
    img_canvas2 = rasterize_dets2(dets2, dims)

    # kwplot.imshow(frame1)

    # img_hmap1 = kwimage.Heatmap.random(shape, noise=0.1, smooth_k=15, ensure_background=False, dets=dets1)
    # img_canvas1 = img_hmap1.draw_on(channel='class_probs').clip(0, 1)

    # img_hmap2 = kwimage.Heatmap.random(shape, noise=0.2, smooth_k=11, ensure_background=False, dets=dets2)
    # img_canvas2 = img_hmap2.draw_on(channel='class_probs').clip(0, 1)

    model1_preds.append(img_canvas1)
    model2_preds.append(img_canvas2)

enemble1_stack = kwimage.stack_images(model1_preds, axis=1, pad=10, bg_value=(1, 1, 1))
enemble2_stack = kwimage.stack_images(model2_preds, axis=1, pad=10, bg_value=(1, 1, 1))

final_stack = (enemble1_stack + enemble2_stack) / 2

ensemble_stack = kwimage.stack_images([enemble1_stack, enemble2_stack], axis=0, pad=10, bg_value=(1, 1, 1))
ensemble_stack = kwimage.stack_images([ensemble_stack, final_stack], axis=0, pad=30, bg_value=(1, 1, 1))
kwplot.imshow(ensemble_stack)
