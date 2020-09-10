import os
import cv2
import numpy as np
import config as c
import tensorflow as tf
import xml.etree.ElementTree as ET
from utils.aug_utils import random_reduce, random_hsv, random_flip, random_crop, random_pca, color_normalize
from utils.anchor_utils import from_box_to_offset, generate_anchors


def load_list(list_path, root_path=c.root_path):
    images = []
    annotations = []
    for dataset in list_path:
        with open(os.path.join(root_path, dataset), 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                images.append(os.path.join(root_path,
                                           dataset.split('/')[0],
                                           'JPEGImages/{}.jpg'.format(line)))
                annotations.append(os.path.join(root_path,
                                                dataset.split('/')[0],
                                                'Annotations/{}.xml'.format(line)))
    return images, annotations


def load_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    boxes = []
    labels = []
    for object in root.iter('object'):
        if object.find('difficult').text == '1':
            continue
        # each object
        # x-width, y-height
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)  # left
        ymin = int(bbox.find('ymin').text)  # top
        xmax = int(bbox.find('xmax').text)  # right
        ymax = int(bbox.find('ymax').text)  # bottom
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(c.class_dict[object.find('name').text])
    return width, height, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.float32)


def load_data(image_path, annotation_path, augment=False):
    image = cv2.imread(image_path.numpy().decode()).astype(np.float32)  # BGR
    _, _, boxes, labels = load_annotation(annotation_path.numpy().decode())

    if augment:
        if np.random.rand() > 0.5:
            image, boxes = random_reduce(image, boxes)
        else:
            image, boxes, labels = random_crop(image, boxes, labels)
        image, boxes = random_flip(image, boxes)
        image = random_hsv(image)
        image = random_pca(image)

    image = color_normalize(image)
    height, width, _ = np.shape(image)
    image = cv2.resize(image, c.input_shape[0: 2])
    boxes[:, 0] = boxes[:, 0] * c.input_shape[1] / width  # left
    boxes[:, 1] = boxes[:, 1] * c.input_shape[0] / height  # top
    boxes[:, 2] = boxes[:, 2] * c.input_shape[1] / width  # right
    boxes[:, 3] = boxes[:, 3] * c.input_shape[0] / height  # bottom

    # anchors = generate_anchors()
    anchors = np.load(c.anchor_cache)
    anchors_offset, anchors_label = from_box_to_offset(boxes, labels, anchors)
    return image, anchors_offset, anchors_label


def load_data_for_test(image_path, annotation_path):
    image = cv2.imread(image_path.numpy().decode()).astype(np.float32)
    image = cv2.resize(image, c.input_shape[0: 2])
    image = color_normalize(image)
    return image, annotation_path


def get_train_data_iterator(list_path=c.train_list_path):
    data_list = load_list(list_path)
    anchors = generate_anchors()
    np.save(c.anchor_cache, anchors)  # anchor cache

    print('Train sample number: {}'.format(len(data_list[0])))
    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.shuffle(len(data_list[0]))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda x, y: tf.py_function(load_data, inp=[x, y, True], Tout=[tf.float32, tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it


def get_test_data_iterator(list_path=c.eval_list_path):
    data_list = load_list(list_path)
    anchors = generate_anchors()
    np.save(c.anchor_cache, anchors)  # anchor cache

    print('Test sample number: {}'.format(len(data_list[0])))
    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.map(lambda x, y: tf.py_function(load_data_for_test, inp=[x, y], Tout=[tf.float32, tf.string]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(c.batch_size)
    it = dataset.__iter__()
    return it


if __name__ == '__main__':
    from utils.eval_utils import show_box
    # show the ground truth

    index = '000046'
    image_path = '/home/user/Documents/dataset/VOC/VOC2007/JPEGImages/{}.jpg'.format(index)
    annotation_path = '/home/user/Documents/dataset/VOC/VOC2007/Annotations/{}.xml'.format(index)
    image = cv2.imread(image_path)
    _, _, boxes, labels = load_annotation(annotation_path)

    for box in boxes:
        show_box(image, box)

    # for anchors visualization
    # use generate_anchors() instead of np.load(c.anchor_cache)
    # And Annotate the 'color normalize' function

    # it = get_train_data_iterator()
    # data = it.next()
    # images, anchors_offset, anchors_labels = data
    #
    # for img, offset, label in zip(images.numpy(), anchors_offset.numpy(), anchors_labels.numpy()):
    #     anchor = generate_anchors()
    #     for i in np.where(label > 0)[0]:
    #         print(c.class_list[int(label[i])])
    #         show_box(img, anchor[i])
