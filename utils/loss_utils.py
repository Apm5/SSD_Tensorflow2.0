import tensorflow as tf
import config as c


def cross_entropy(cls_true, cls_pred, from_logits=False):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=cls_true, y_pred=cls_pred, from_logits=from_logits)
    loss = tf.reduce_mean(loss)
    return loss


def smooth_l1(loc_true, loc_pred):
    diff = tf.abs(loc_true - loc_pred)
    smooth_l1_option1 = 0.5 * diff * diff
    smooth_l1_option2 = diff - 0.5
    loss = tf.where(tf.less(diff, 1.0), smooth_l1_option1, smooth_l1_option2)
    loss = tf.reduce_mean(loss)
    return loss


def l2_loss(model, weight=c.weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name or 'bias' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weight


def cal_loss(cls_true, loc_true, cls_pred, loc_pred):
    """
    In cls_true, -1 is the ignore mark.
    Args:
        cls_true: [batch, anchor num]
        loc_true: [batch, anchor num, 4]
        cls_pred: [batch, anchor num, class num]
        loc_pred: [batch, anchor num, 4]

    Returns:

    """

    positive_num = tf.math.count_nonzero(tf.greater(cls_true, 0), axis=-1)
    negative_num = tf.math.count_nonzero(tf.equal(cls_true, 0), axis=-1)
    negative_select_num = tf.math.minimum(negative_num, positive_num * c.hard_mining_ratio)

    positive_mask = tf.greater(cls_true, 0)

    # hard negative mining for classification
    negative_mask = tf.equal(cls_true, 0)
    bg_pred = cls_pred[:, :, 0]
    bg_pred_for_negative = tf.where(negative_mask,
                                    0.0 - bg_pred,
                                    0.0 - tf.ones_like(bg_pred))  # ignore the positive anchors
    topk_bg_pred, _ = tf.nn.top_k(bg_pred_for_negative, k=tf.shape(bg_pred_for_negative)[1])
    topk_threshold = tf.gather_nd(topk_bg_pred,
                                  tf.stack([tf.range(c.batch_size, dtype=tf.int64), negative_select_num - 1],
                                           axis=-1))
    negative_mask = tf.greater_equal(bg_pred_for_negative, tf.expand_dims(topk_threshold, axis=-1))

    mask = tf.logical_or(positive_mask, negative_mask)

    flaten_cls_true_masked = tf.reshape(tf.boolean_mask(cls_true, mask), [-1])
    flaten_cls_pred_masked = tf.reshape(tf.boolean_mask(cls_pred, mask), [-1, c.class_num])

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(flaten_cls_true_masked, dtype=tf.int32),
                                               tf.cast(tf.argmax(flaten_cls_pred_masked, axis=-1), dtype=tf.int32)),
                                      dtype=tf.float32))

    # mean for positive anchor num
    cls_loss = cross_entropy(flaten_cls_true_masked, flaten_cls_pred_masked) * (c.hard_mining_ratio + 1)

    flaten_loc_true_masked = tf.reshape(tf.boolean_mask(loc_true, positive_mask), [-1, 4])
    flaten_loc_pred_masked = tf.reshape(tf.boolean_mask(loc_pred, positive_mask), [-1, 4])

    # mean for positive anchor num
    loc_loss = smooth_l1(flaten_loc_true_masked, flaten_loc_pred_masked) * 4

    return accuracy, cls_loss, loc_loss
