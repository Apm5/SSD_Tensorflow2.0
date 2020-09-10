import tensorflow as tf
import numpy as np
import config as c
import os
import cv2
import copy
from model.SSD import SSD
from utils.eval_utils import show_box
from utils.aug_utils import color_normalize
from utils.anchor_utils import generate_anchors, from_offset_to_box
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_path = c.weight_file
image_path = '/home/user/Documents/dataset/VOC/VOC2007TEST/JPEGImages/000388.jpg'


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    model = SSD()
    model.build(input_shape=(None,) + c.input_shape)
    model.summary()
    model.load_weights(model_path)

    image = cv2.imread(image_path)
    height, width, _ = np.shape(image)
    input_image = np.array([color_normalize(cv2.resize(copy.copy(image), tuple(c.input_shape[:2])))], dtype=np.float32)
    cls_pred, loc_pred = model(input_image, training=False)

    anchors = generate_anchors()
    boxes, scores, labels = from_offset_to_box(loc_pred[0], cls_pred[0], anchors, score_threshold=0.1)

    for box, score, label in zip(boxes, scores, labels):
        box[0] = box[0] / c.input_shape[1] * width  # left
        box[1] = box[1] / c.input_shape[0] * height  # top
        box[2] = box[2] / c.input_shape[1] * width  # right
        box[3] = box[3] / c.input_shape[0] * height  # bottom
        print('image: {}\nclass: {}\nconfidence: {:.4f}\n'.format(image_path, c.class_list[label], score))
        show_box(image, box)
