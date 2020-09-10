import cv2
import numpy as np
import config as c


def random_reduce(image, boxes):
    # image will be resize to input_shape
    # reduce image by padding border
    height, width, _ = np.shape(image)
    ratio = np.random.uniform(c.reduce_min, 1.0)
    new_height, new_width = int(height / ratio), int(width / ratio)
    position_x = np.random.randint(0, new_width - width + 1)
    position_y = np.random.randint(0, new_height - height + 1)
    image = cv2.copyMakeBorder(image,
                               top=position_y,
                               bottom=new_height - position_y - height,
                               left=position_x,
                               right=new_width - position_x - width,
                               borderType=cv2.BORDER_CONSTANT,
                               value=c.mean)  # for normalize
    boxes = boxes + np.array([position_x, position_y, position_x, position_y])

    return image, boxes


def random_crop(image, boxes, labels, max_repeat=10):
    height, width, _ = np.shape(image)
    center_boxes = np.zeros([len(boxes), 2])  # [width_center, height_center]
    center_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    center_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2

    for _ in range(max_repeat):
        crop_height = int(np.random.uniform(c.crop_min, 1.0) * height)
        crop_width = int(np.random.uniform(c.crop_min, 1.0) * width)

        crop_height = np.minimum(crop_height, crop_width * c.crop_aspect_max)
        crop_width = np.minimum(crop_width, crop_height * c.crop_aspect_max)

        crop_l = np.random.randint(0, width - crop_width)
        crop_t = np.random.randint(0, height - crop_height)

        crop_box = np.array([crop_l, crop_t, crop_l + crop_width, crop_t + crop_height], dtype=np.int32)

        # object center in crop box
        boxes_in_crop = np.logical_and(np.logical_and(center_boxes[:, 0] > crop_box[0], center_boxes[:, 0] < crop_box[2]),
                                       np.logical_and(center_boxes[:, 1] > crop_box[1], center_boxes[:, 1] < crop_box[3]))

        # at least 1 object
        if np.sum(boxes_in_crop) > 1:
            image = image[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2], :]
            boxes_in_crop_box = []
            labels_in_crop_box = []
            for i, box in enumerate(boxes):
                if boxes_in_crop[i]:
                    boxes_in_crop_box.append([np.maximum(box[0] - crop_box[0], 0),
                                              np.maximum(box[1] - crop_box[1], 0),
                                              np.minimum(box[2] - crop_box[0], crop_box[2]),
                                              np.minimum(box[3] - crop_box[1], crop_box[3])])
                    labels_in_crop_box.append(labels[i])
            return image, np.array(boxes_in_crop_box, dtype=np.float32), np.array(labels_in_crop_box, dtype=np.float32)

    # try max_repeat times
    # return original image
    return image, boxes, labels


def random_flip(image, boxes):
    height, width, _ = np.shape(image)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    return image, boxes


def random_hsv(image):
    random_h = np.random.uniform(c.hue_delta[0], c.hue_delta[1])
    random_s = np.random.uniform(c.saturation_scale[0], c.saturation_scale[1])
    random_v = np.random.uniform(c.brightness_scale[0], c.brightness_scale[1])

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = image_hsv[:, :, 0] + random_h % 360.0  # hue
    image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * random_s, 1.0)  # saturation
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * random_v, 255.0)  # brightness

    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)


def random_pca(image):
    alpha = np.random.normal(0, c.pca_std, size=(3,))
    offset = np.dot(c.eigvec * alpha, c.eigval)
    image = image + offset
    return np.maximum(np.minimum(image, 255.0), 0.0)


def color_normalize(image):
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - c.mean[i])# / c.std[i]
    return image
