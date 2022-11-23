"""
Proof of concept that uses an mmdetection head given a custom feature map.
"""
from mmdet.models.dense_heads.centernet_head import CenterNetHead
import torch
import mmcv
import collections
import kwimage

CenterNetHeadOutput = collections.namedtuple('CenterNetHeadOutput', ['center_heatmap_preds', 'wh_preds', 'offset_preds'])


def _dummy_img_meta(H, W, C):
    return {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'border': (0, 0, 0, 0),
        'batch_input_shape': (H, W),
        'filename': '<memory>.png',
        'scale_factor': [1.0, 1.0, 1.0, 1.0],
        'flip': False,
    }

# batch size, #classes, width, height
B, C, W, H = 3, 11, 256, 256

input_feats = 5
hidden_feats = 7

# Build the head
head = CenterNetHead(
    input_feats, hidden_feats, C,
    # The test cfg is required to decode boxes
    test_cfg=mmcv.Config({'topk': 7, 'local_maximum_kernel': 3}),
)

# Build random features
features = torch.rand(B, input_feats, H, W)

# mmdet expects a pyramid of inputs, so use a single level
level_features = [features]


# Get predictions from the features
raw_output = head.forward(level_features)
raw_output = CenterNetHeadOutput(*raw_output)


# Construct dummy truth
gt_bboxes = []
gt_labels = []
img_metas = []
gt_bboxes_ignore = []

for bx in range(B):
    # Use kwimage to make dummy truth for each batch item
    true_dets = kwimage.Detections.random(
        num=10, classes=C, rng=0).scale((W, H)).tensor()
    # Construct the raw tensors for mmdet
    bboxes = true_dets.boxes.to_ltrb().data
    labels = true_dets.class_idxs
    ignores = (true_dets.scores <= 0)  # use score as a proxy for ignore
    img_meta = _dummy_img_meta(H, W, C)
    gt_bboxes.append(bboxes)
    gt_labels.append(labels)
    img_metas.append(img_meta)
    gt_bboxes_ignore.append(ignores)


# Compute the loss
loss = head.loss(
    center_heatmap_preds=raw_output.center_heatmap_preds,
    wh_preds=raw_output.wh_preds,
    offset_preds=raw_output.offset_preds,
    gt_bboxes=gt_bboxes,
    gt_labels=gt_labels,
    img_metas=img_metas,
    gt_bboxes_ignore=gt_bboxes_ignore
)

# Decode the box representation
mm_boxes_batch = head.get_bboxes(
    center_heatmap_preds=raw_output.center_heatmap_preds,
    wh_preds=raw_output.wh_preds,
    offset_preds=raw_output.offset_preds,
    img_metas=img_metas,
    rescale=True,
    with_nms=False,
)

# Turn the mm boxes into a kwimage representation
batch_dets = []
for mm_boxes in mm_boxes_batch:
    pred_ltrbs, pred_cidx = mm_boxes
    pred_boxes = kwimage.Boxes(pred_ltrbs[:, 0:4], 'ltrb')
    pred_scores = pred_ltrbs[:, 4]

    pred_dets = kwimage.Detections(
        boxes=pred_boxes,
        class_idxs=pred_cidx,
        scores=pred_scores
    )
    pred_dets = pred_dets.numpy().non_max_supress()
    batch_dets.append(pred_dets)

# show
import kwplot
kwplot.autompl()
batch_dets[0].draw(setlim=1)
