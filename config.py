# Training config
batch_size = 32
input_shape = (300, 300, 3)  # (height, width, channel)
weight_decay = 5e-4

train_num = 5011 + 11540  # 16551
test_num = 4952
iterations_per_epoch = int(train_num / batch_size)
test_iterations = int(test_num / batch_size) + 1

initial_learning_rate = 0.001
minimum_learning_rate = 0.00001
warm_up_step = 1000
epoch_num = 100

root_path = '/home/user/Documents/dataset/VOC'
train_list_path = ['VOC2007/ImageSets/Main/trainval.txt',  # 5011
                   'VOC2012/ImageSets/Main/trainval.txt']  # 11540
eval_list_path = ['VOC2007TEST/ImageSets/Main/test.txt']  # 4952

pretrain_weight = 'weights/vgg16_weights_for_SSD.h5'
log_file = 'result/log/SSD_warm_up.txt'
weight_file = 'result/weight/SSD_warm_up.h5'
det_result_file = 'result/det_result'  # save the result for VOC eval tool

# Augmentation config
# From 'Bag of tricks for image classification with convolutional neural networks'
# Or https://github.com/dmlc/gluon-cv

reduce_min = 0.5
crop_min = 0.5
crop_aspect_max = 2.0

hue_delta = (-36, 36)
saturation_scale = (0.6, 1.4)
brightness_scale = (0.6, 1.4)
pca_std = 0.1

mean = [103.939, 116.779, 123.68]  # BGR format
std = [58.393, 57.12, 57.375]
eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]

# Anchor config
anchor_cache = 'utils/anchors.npy'
anchor_scale = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
anchor_num_each_pixel = [4, 6, 6, 6, 4, 4]
feature_map_size = [38, 19, 10, 5, 3, 1]
anchor_num_each_scale = [anchor_num_each_pixel[i] * feature_map_size[i] ** 2 for i in range(6)]
prior_scaling = [0.1, 0.1, 0.2, 0.2]

anchor_positive_threshold = 0.5  # anchor is positive if (iou(anchor, bbox) > threshold)
anchor_negative_threshold = 0.5  # anchor is negative if (iou(anchor, bbox) < threshold)
hard_mining_ratio = 3  # negative : positive

# Evaluation config
nms_max_box_num = 50
nms_iou_threshold = 0.45
box_size_threshold = 10  # ignore the small output box
score_threshold = 0.01  # box is positive if score > score_threshold
anchor_belongs_to_one_class = False  # one anchor belongs to only one class

# VOC dataset config
class_num = 20 + 1  # '0' for back ground

class_dict = {'back_ground': 0,
              'aeroplane': 1,
              'bicycle': 2,
              'bird': 3,
              'boat': 4,
              'bottle': 5,
              'bus': 6,
              'car': 7,
              'cat': 8,
              'chair': 9,
              'cow': 10,
              'diningtable': 11,
              'dog': 12,
              'horse': 13,
              'motorbike': 14,
              'person': 15,
              'pottedplant': 16,
              'sheep': 17,
              'sofa': 18,
              'train': 19,
              'tvmonitor': 20}

class_list = ['back_ground',
              'aeroplane',
              'bicycle',
              'bird',
              'boat',
              'bottle',
              'bus',
              'car',
              'cat',
              'chair',
              'cow',
              'diningtable',
              'dog',
              'horse',
              'motorbike',
              'person',
              'pottedplant',
              'sheep',
              'sofa',
              'train',
              'tvmonitor']
