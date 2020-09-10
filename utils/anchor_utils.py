import numpy as np
import config as c
import tensorflow as tf


def from_xywh_to_ltrb(xywh_box):
    ltrb_box = np.zeros(4, dtype=np.float32)
    ltrb_box[0] = xywh_box[0] - xywh_box[2] / 2
    ltrb_box[1] = xywh_box[1] - xywh_box[3] / 2
    ltrb_box[2] = xywh_box[0] + xywh_box[2] / 2
    ltrb_box[3] = xywh_box[1] + xywh_box[3] / 2
    return ltrb_box


def from_ltrb_to_xywh(ltrb_box):
    xywh_box = np.zeros(4, dtype=np.float32)
    xywh_box[0] = (ltrb_box[0] + ltrb_box[2]) / 2
    xywh_box[1] = (ltrb_box[1] + ltrb_box[3]) / 2
    xywh_box[2] = ltrb_box[2] - ltrb_box[0]
    xywh_box[3] = ltrb_box[3] - ltrb_box[1]
    return xywh_box


def cal_iou_parallel(anchors, boxes):
    """

    Args:
        anchors: [anchor_num, 4]
        boxes: [ground_truth_num, 4]

    Returns: [anchor_num, ground_truth_num]

    """
    # intersection
    anchors = np.expand_dims(anchors, axis=1)
    boxes = np.expand_dims(boxes, axis=0)

    ixmin = np.maximum(anchors[:, :, 0], boxes[:, :, 0])
    iymin = np.maximum(anchors[:, :, 1], boxes[:, :, 1])
    ixmax = np.minimum(anchors[:, :, 2], boxes[:, :, 2])
    iymax = np.minimum(anchors[:, :, 3], boxes[:, :, 3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = (anchors[:, :, 2] - anchors[:, :, 0] + 1.) * (anchors[:, :, 3] - anchors[:, :, 1] + 1.) +\
          (boxes[:, :, 2] - boxes[:, :, 0] + 1.) * (boxes[:, :, 3] - boxes[:, :, 1] + 1.) -\
          inters

    overlaps = inters / uni
    return overlaps


def cal_iou(a, b):
    """
    box is [xmin, ymin, xmax, ymax]
    Args:
        a: bbox
        b: bbox

    Returns: IoU

    """
    # intersection
    ixmin = np.maximum(a[0], b[0])
    iymin = np.maximum(a[1], b[1])
    ixmax = np.minimum(a[2], b[2])
    iymax = np.minimum(a[3], b[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = (a[2] - a[0] + 1.) * (a[3] - a[1] + 1.) +\
          (b[2] - b[0] + 1.) * (b[3] - b[1] + 1.) -\
          inters

    overlaps = inters / uni
    return overlaps


def generate_anchors():
    anchors = []

    for i, (num, size) in enumerate(zip(c.anchor_num_each_pixel, c.feature_map_size)):
        # kth feature map, number of boxes in one pixel and feature map size
        # [38, 19, 10, 5, 3, 1]
        for y in range(size):
            for x in range(size):
                # position (x, y) in feature map
                # above order is important.
                # in math width (x) comes first
                # in data structure height (y) comes first
                center_x = (x + 0.5) / size
                center_y = (y + 0.5) / size
                if num == 4:
                    aspect_ratio = [1, 1, 2, 0.5]
                    scale = np.full([4], c.anchor_scale[i])
                    scale[1] = np.sqrt(c.anchor_scale[i] * c.anchor_scale[i + 1])
                if num == 6:
                    aspect_ratio = [1, 1, 2, 0.5, 3, 1 / 3.0]
                    scale = np.full([6], c.anchor_scale[i])
                    scale[1] = np.sqrt(c.anchor_scale[i] * c.anchor_scale[i + 1])

                for box_index, (ar, s) in enumerate(zip(aspect_ratio, scale)):
                    w = s * np.sqrt(ar)
                    h = s / np.sqrt(ar)

                    anchor = [(center_x - w / 2) * c.input_shape[1],
                              (center_y - h / 2) * c.input_shape[0],
                              (center_x + w / 2) * c.input_shape[1],
                              (center_y + h / 2) * c.input_shape[0]]

                    anchors.append(anchor)
    return np.array(anchors, dtype=np.float32)


def from_box_to_offset(boxes, boxes_label, anchors,
                       positive_threshold=c.anchor_positive_threshold,
                       negative_threshold=c.anchor_negative_threshold):
    """
    According to the ground truth, generate the offset and label for each anchor.

    Args:
        boxes: ground truth bounding box, [boxes num, 4]
        boxes_label: labels of boxes, [boxes_num]
        anchors: default anchors, [anchors_num, 4]
        positive_threshold: anchor is positive if (iou(anchor, bbox) > threshold)
        negative_threshold: anchor is negative if (iou(anchor, bbox) < threshold)

    Returns: anchor_offset, anchor_labels

    """
    anchors_offset = np.zeros_like(anchors)
    anchors_label = np.zeros([np.sum(c.anchor_num_each_scale)], dtype=np.float32)

    anchors_iou = cal_iou_parallel(anchors, boxes)  # [anchor num, ground truth num]
    anchors_max_iou = np.max(anchors_iou, axis=-1)
    positive_index = np.where(anchors_max_iou > positive_threshold)[0]

    for anchor_index in positive_index:
        match_gt_index = np.argmax(anchors_iou[anchor_index])
        anchors_label[anchor_index] = boxes_label[match_gt_index]

        anchor = anchors[anchor_index]
        gx, gy, gw, gh = from_ltrb_to_xywh(boxes[match_gt_index])  # g for ground truth
        ax, ay, aw, ah = from_ltrb_to_xywh(anchor)  # a for anchor

        anchors_offset[anchor_index, 0] = (gx - ax) / aw / c.prior_scaling[0]
        anchors_offset[anchor_index, 1] = (gy - ay) / ah / c.prior_scaling[1]
        anchors_offset[anchor_index, 2] = np.log(gw / aw) / c.prior_scaling[2]
        anchors_offset[anchor_index, 3] = np.log(gh / ah) / c.prior_scaling[3]

    ignore_index = np.where(np.logical_and(anchors_max_iou > negative_threshold,
                                           anchors_max_iou < positive_threshold))[0]
    for anchor_index in ignore_index:
        anchors_label[anchor_index] = -1

    return anchors_offset, anchors_label


def from_offset_to_box(anchors_offset, anchors_score, anchors, anchor_belongs_to_one_class=c.anchor_belongs_to_one_class,
                       score_threshold=c.score_threshold, box_size_threshold=c.box_size_threshold,
                       nms_max_box_num=c.nms_max_box_num, nms_iou_threshold=c.nms_iou_threshold):
    """
    According to the confidence threshold and NMS, the non object anchor is filtered out. Normally, each anchor
    should belong to only one class. However, the model can get better result on mAP evaluation metrics by
    setting 'anchor_belongs_to_one_class' to True.

    Args:
        anchors_offset: model output, [anchor num, 4]
        anchors_score: model output, [anchor num, class num]
        anchors: default anchors, [anchors_num, 4]
        anchor_belongs_to_one_class: one anchor belongs to only one class
        score_threshold: box is positive if score > score_threshold
        box_size_threshold: ignore the small output box
        nms_max_box_num:
        nms_iou_threshold:

    Returns: boxes, scores, labels

    """
    boxes = []
    scores = []
    labels = []

    anchors_label = np.argmax(anchors_score, axis=-1)
    for cls_index in range(1, c.class_num):  # ignore back ground
        anchor_index_list = np.where(anchors_score[:, cls_index] > score_threshold)[0]
        cls_boxes = []
        cls_scores = []
        for anchor_index in anchor_index_list:
            if anchor_belongs_to_one_class and anchors_label[anchor_index] != cls_index:
                continue
            offset = anchors_offset[anchor_index]
            ax, ay, aw, ah = from_ltrb_to_xywh(anchors[anchor_index])
            x = offset[0] * c.prior_scaling[0] * aw + ax
            y = offset[1] * c.prior_scaling[1] * ah + ay
            w = np.exp(offset[2] * c.prior_scaling[2]) * aw
            h = np.exp(offset[3] * c.prior_scaling[3]) * ah

            box = from_xywh_to_ltrb([x, y, w, h])
            box[0] = np.maximum(box[0], 0)
            box[1] = np.maximum(box[1], 0)
            box[2] = np.minimum(box[2], c.input_shape[1])
            box[3] = np.minimum(box[3], c.input_shape[0])

            if box[2] - box[0] > box_size_threshold and box[3] - box[1] > box_size_threshold:  # ignore the small box
                cls_boxes.append(box)
                cls_scores.append(anchors_score[anchor_index, cls_index])

        if len(cls_boxes) > 0:
            nms_index = tf.image.non_max_suppression(boxes=cls_boxes,
                                                     scores=cls_scores,
                                                     max_output_size=nms_max_box_num,
                                                     iou_threshold=nms_iou_threshold).numpy()

            for box, score in zip(np.array(cls_boxes)[nms_index], np.array(cls_scores)[nms_index]):
                boxes.append(box)
                scores.append(score)
                labels.append(cls_index)
    return np.array(boxes), scores, labels


if __name__ == '__main__':
    anchors = generate_anchors()
    print(np.shape(anchors))
    np.save('anchors.npy', anchors)
