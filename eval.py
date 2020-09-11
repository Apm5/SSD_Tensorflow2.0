import os
import tensorflow as tf
import config as c
from tqdm import tqdm
from model.SSD import SSD
from utils.anchor_utils import from_offset_to_box, generate_anchors
from utils.eval_utils import voc_eval
from utils.data_utils import get_test_data_iterator, load_annotation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@tf.function
def eval_step(model, images):
    cls_pred, loc_pred = model(images, training=False)
    return cls_pred, loc_pred


def eval(model, data_iterator):
    anchors = generate_anchors()
    for _ in tqdm(range(c.test_iterations)):
        batch_images, batch_annotation_path = data_iterator.next()
        batch_cls_pred, batch_loc_pred = eval_step(model, batch_images)

        batch_cls_pred = batch_cls_pred.numpy()
        batch_loc_pred = batch_loc_pred.numpy()
        batch_annotation_path = map(bytes.decode, batch_annotation_path.numpy())

        for cls_pred, loc_pred, annotation_path in zip(batch_cls_pred, batch_loc_pred, batch_annotation_path):
            width, height, _, _ = load_annotation(annotation_path)
            image_id = annotation_path.split('/')[-1][: -4]  # '.../Annotations/123456.xml'
            boxes, scores, labels = from_offset_to_box(loc_pred, cls_pred, anchors)

            # Resize the box to match the original image
            # So the network input image SHOULD only be resized without any other augmentation.
            if len(boxes) > 0:
                boxes[:, 0] = boxes[:, 0] / c.input_shape[1] * width  # left
                boxes[:, 1] = boxes[:, 1] / c.input_shape[0] * height  # top
                boxes[:, 2] = boxes[:, 2] / c.input_shape[1] * width  # right
                boxes[:, 3] = boxes[:, 3] / c.input_shape[0] * height  # bottom

            for box, score, label in zip(boxes, scores, labels):
                with open(os.path.join(c.det_result_file, '{}.txt').format(c.class_list[label]), 'a') as f:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(image_id,
                                                                               score,
                                                                               *box))

    mAP = 0
    for cls_name in c.class_list[1:]:  # ignore back ground
        rec, prec, ap = voc_eval(detpath=os.path.join(c.det_result_file, '{}.txt'),
                                 annopath=os.path.join(c.root_path, c.eval_list_path[0].split('/')[0], 'Annotations/{}.xml'),
                                 imagesetfile=os.path.join(c.root_path, c.eval_list_path[0]),
                                 classname=cls_name)

        print('{} AP = {:.4f}, rec = {:.4f}, prec = {:.4f}'.format(cls_name, ap, rec[-1], prec[-1]))
        mAP += ap
    mAP /= c.class_num - 1
    print('Mean AP = {:.4f}'.format(mAP))


def clear_det_result(det_result_file=c.det_result_file):
    for file in os.listdir(det_result_file):
        os.remove(os.path.join(det_result_file, file))


if __name__ == '__main__':
    # # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    test_data_iterator = get_test_data_iterator(c.eval_list_path)

    # get model
    model = SSD()
    model.build(input_shape=(None,) + c.input_shape)
    model.summary()
    model.load_weights(c.weight_file)

    clear_det_result()
    eval(model, test_data_iterator)
