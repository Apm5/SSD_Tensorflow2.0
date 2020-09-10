import os
import config as c
import tensorflow as tf
from model.SSD import SSD
from tqdm import tqdm
from utils.data_utils import get_train_data_iterator
from utils.loss_utils import cal_loss, l2_loss
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)


@tf.function
def train_step(model, images, cls_true, loc_true, optimizer):
    with tf.GradientTape() as tape:
        cls_pred, loc_pred = model(images, training=True)
        cls_accuracy, cls_loss, loc_loss = cal_loss(cls_true, loc_true, cls_pred, loc_pred)
        loss = cls_loss + loc_loss + l2_loss(model)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return cls_accuracy, cls_loss, loc_loss


@tf.function
def test_step(model, images, cls_true, loc_true):
    cls_pred, loc_pred = model(images, training=False)
    cls_accuracy, cls_loss, loc_loss = cal_loss(cls_true, loc_true, cls_pred, loc_pred)
    return cls_accuracy, cls_loss, loc_loss


def train(model, optimizer, data_iterator, log_file):
    sum_c_loss = 0
    sum_l_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(c.iterations_per_epoch)):
        images, anchors_offset, anchors_label = data_iterator.next()
        accuracy, c_loss, l_loss = train_step(model, images, anchors_label, anchors_offset, optimizer)

        print('accuracy: {:.4f}, class loss: {:.4f}, location loss: {:.4f}, l2: {:.4f}'.format(accuracy,
                                                                                               c_loss,
                                                                                               l_loss,
                                                                                               l2_loss(model)))

        sum_c_loss += c_loss
        sum_l_loss += l_loss
        sum_accuracy += accuracy

    log_file.write('accuracy: {:.4f}, class loss: {:.4f}, location loss: {:.4f}, l2: {:.4f}\n'.format(sum_accuracy / c.iterations_per_epoch,
                                                                                                      sum_c_loss / c.iterations_per_epoch,
                                                                                                      sum_l_loss / c.iterations_per_epoch,
                                                                                                      l2_loss(model)))


def test(model, data_iterator, log_file):
    sum_c_loss = 0
    sum_l_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(c.test_iterations)):
        images, anchors_offset, anchors_label = data_iterator.next()
        accuracy, c_loss, l_loss = test_step(model, images, anchors_label, anchors_offset)

        print('accuracy: {:.4f}, class loss: {:.4f}, location loss: {:.4f}'.format(accuracy,
                                                                                   c_loss,
                                                                                   l_loss))

        sum_c_loss += c_loss
        sum_l_loss += l_loss
        sum_accuracy += accuracy

    log_file.write('accuracy: {:.4f}, class loss: {:.4f}, location loss: {:.4f}\n'.format(sum_accuracy / c.test_iterations,
                                                                                          sum_c_loss / c.test_iterations,
                                                                                          sum_l_loss / c.test_iterations))


if __name__ == '__main__':
    # # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    train_data_iterator = get_train_data_iterator()

    # get model
    model = SSD()

    # show
    model.build(input_shape=(None, ) + c.input_shape)
    model.summary()
    model.load_weights(c.pretrain_weight, by_name=True)

    # train
    learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=c.initial_learning_rate,
                                                    decay_steps=c.epoch_num * c.iterations_per_epoch,
                                                    alpha=c.minimum_learning_rate,
                                                    warm_up_step=c.warm_up_step)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
    for epoch_num in range(c.epoch_num):
        with open(c.log_file, 'a') as f:
            f.write('epoch:{}\n'.format(epoch_num))
            train(model, optimizer, train_data_iterator, f)
            if epoch_num % 5 == 4:
                f.write('test:\n')
                test_data_iterator = get_train_data_iterator(c.eval_list_path)
                test(model, test_data_iterator, f)
        model.save_weights(c.weight_file, save_format='h5')
