import tensorflow as tf
import config as c
from utils.loss_utils import l2_loss
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D


class L2Normalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=input_shape[-1],
                                     initializer=tf.initializers.constant(value=20.0),
                                     trainable=True)

    def call(self, inputs):
        square_sum = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        inv_norm = tf.math.rsqrt(tf.maximum(square_sum, 1e-7))
        norm = tf.multiply(inputs, inv_norm)
        output = tf.multiply(norm, self.scale)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class SSD(models.Model):
    def __init__(self, **kwargs):
        super(SSD, self).__init__(**kwargs)
        # Block 1
        self.conv1_1 = Conv2D(64, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block1_conv1')

        self.conv1_2 = Conv2D(64, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block1_conv2')

        # Block 2
        self.conv2_1 = Conv2D(128, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block2_conv1')
        self.conv2_2 = Conv2D(128, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block2_conv2')

        # Block 3
        self.conv3_1 = Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block3_conv1')
        self.conv3_2 = Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block3_conv2')
        self.conv3_3 = Conv2D(256, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block3_conv3')

        # Block 4
        self.conv4_1 = Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv1')
        self.conv4_2 = Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv2')
        self.conv4_3 = Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block4_conv3')
        self.l2normalization = L2Normalization()

        # Block 5
        self.conv5_1 = Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block5_conv1')
        self.conv5_2 = Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block5_conv2')
        self.conv5_3 = Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='block5_conv3')

        # VGG fc1 and fc2 >> SSD conv6 and conv7
        self.conv6_1 = Conv2D(1024, (3, 3),
                              activation='relu',
                              padding='same',
                              name='fc6',
                              dilation_rate=(3, 3))  # use atrous

        self.conv7_1 = Conv2D(1024, (1, 1),
                              activation='relu',
                              padding='same',
                              name='fc7')

        self.conv8_1 = Conv2D(256, (1, 1),
                              activation='relu',
                              padding='same',
                              name='block8_conv1')

        self.conv8_2 = Conv2D(512, (3, 3),
                              activation='relu',
                              strides=(2, 2),
                              padding='same',
                              name='block8_conv2')

        self.conv9_1 = Conv2D(128, (1, 1),
                              activation='relu',
                              padding='same',
                              name='block9_conv1')

        self.conv9_2 = Conv2D(256, (3, 3),
                              activation='relu',
                              strides=(2, 2),
                              padding='same',
                              name='block9_conv2')

        self.conv10_1 = Conv2D(128, (1, 1),
                               activation='relu',
                               padding='same',
                               name='block10_conv1')

        self.conv10_2 = Conv2D(256, (3, 3),
                               activation='relu',
                               padding='valid',
                               name='block10_conv2')

        self.conv11_1 = Conv2D(128, (1, 1),
                               activation='relu',
                               padding='same',
                               name='block11_conv1')

        self.conv11_2 = Conv2D(256, (3, 3),
                               activation='relu',
                               padding='valid',
                               name='block11_conv2')

        self.cls_collector = []
        self.loc_collector = []
        for i in range(6):
            self.cls_collector.append(Conv2D(c.anchor_num_each_pixel[i] * c.class_num, (3, 3),
                                             padding='same',
                                             name='cls_{}'.format(i)))
            self.loc_collector.append(Conv2D(c.anchor_num_each_pixel[i] * 4, (3, 3),
                                             padding='same',
                                             name='loc_{}'.format(i)))

    def call(self, inputs):
        feature_collector = []

        net = self.conv1_1(inputs)
        net = self.conv1_2(net)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv2_1(net)
        net = self.conv2_2(net)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv3_1(net)
        net = self.conv3_2(net)
        net = self.conv3_3(net)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv4_1(net)
        net = self.conv4_2(net)
        net = self.conv4_3(net)

        feature_collector.append(self.l2normalization(net))
        # print(net.shape)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv5_1(net)
        net = self.conv5_2(net)
        net = self.conv5_3(net)
        net = tf.nn.max_pool2d(net, ksize=(3, 3), strides=(1, 1), padding='SAME')

        net = self.conv6_1(net)
        net = self.conv7_1(net)
        feature_collector.append(net)
        # print(net.shape)

        net = self.conv8_1(net)
        net = self.conv8_2(net)
        feature_collector.append(net)
        # print(net.shape)

        net = self.conv9_1(net)
        net = self.conv9_2(net)
        feature_collector.append(net)
        # print(net.shape)

        net = self.conv10_1(net)
        net = self.conv10_2(net)
        feature_collector.append(net)
        # print(net.shape)

        net = self.conv11_1(net)
        net = self.conv11_2(net)
        feature_collector.append(net)
        # print(net.shape)

        cls_list = []
        loc_list = []
        for i, feature_map in enumerate(feature_collector):
            cls_list.append(tf.reshape(self.cls_collector[i](feature_map), [-1, c.anchor_num_each_scale[i], c.class_num]))
            loc_list.append(tf.reshape(self.loc_collector[i](feature_map), [-1, c.anchor_num_each_scale[i], 4]))
            print('feature map {}'.format(i), feature_map.shape, cls_list[i].shape, loc_list[i].shape)

        cls = tf.concat(cls_list, axis=1)
        cls = tf.nn.softmax(cls)
        loc = tf.concat(loc_list, axis=1)

        print(cls.shape, loc.shape)
        return cls, loc


if __name__ == '__main__':
    model = SSD()
    model.build(input_shape=(None, 300, 300, 3))
    model.summary()
    print(l2_loss(model))

    model.load_weights('../weights/vgg16_weights_for_SSD.h5')
    print(l2_loss(model))
