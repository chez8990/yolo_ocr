import sys
sys.path.append(r'C:\Users\Chester\Desktop\python\yolo_ocr')

import numpy as np
import tensorflow as tf
import keras.backend as K
import utils
# from .. import utils
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, LeakyReLU

class yolo(Model):
    # Create an instance of the yolo model
    # Arguments:
    #   input_dimension: tuple, the dimensions of the input_image
    #   nb_classes: integer, indicates the number of classes YOLO should detect
    #   grid_number: integer, specifies the number of grids in a row.

    def __init__(self,
                 input_dimension,
                 nb_classes,
                 grid_number,
                 lambda_coord=0.5,
                 lambda_obj=0.25,
                 lambda_noobj=0.25):

        self.input_dim = input_dimension
        self.nb_classes = nb_classes
        self.grid_number = grid_number
        self.grid_size = int(input_dimension[0] / grid_number)
        self.lc = lambda_coord
        self.lo = lambda_obj
        self.noj = lambda_noobj

        self.output_dim = grid_number**2 * (5 + nb_classes)

        self.Input = Input(shape=input_dimension)
        self.Output = self._build_layers()

        super().__init__(inputs=self.Input, outputs=self.Output)

    def _build_layers(self):
        Input = self.Input

        x = Conv2D(filters=8, kernel_size=5, strides=2)(Input)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        x = Conv2D(filters=4, kernel_size=5, strides=1)(x)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        # x = Conv2D(filters=64, kernel_size=5, strides=1)(Input)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Conv2D(filters=64, kernel_size=5, strides=1)(Input)
        # x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        x = Dense(self.output_dim)(x)
        x = LeakyReLU(alpha=0.1)(x)

        return x

    def yolo_loss(self, y_true, y_pred):
        S = self.grid_number
        C = self.nb_classes

        # y_true will be of the shape (None, S, S ,15), we are unwrapping
        obj_presence = y_true[:, S * S]
        obj_not_presence = 1 - obj_presence
        obj_coord = tf.reshape(y_true[:, S * S:S * S * 3], [-1, S*S, 2])
        obj_hw = tf.reshape(y_true[:, S * S * 3:S * S * 5],  [-1, S*S, 2])
        obj_class = tf.reshape(y_true[:, S * S * 5:], [-1, S*S, C])

        # Do the same for y_pred
        print(y_pred)
        obj_presence_hat = tf.reshape(y_pred[:, S * S], [-1, S*S])
        obj_coord_hat = tf.reshape(y_pred[:, S * S:S * S * 3], [-1, S*S, 2])
        obj_hw_hat = tf.exp(0.5 * tf.reshape(y_pred[:, S * S * 3:S * S * 5], [-1, S*S, 2]))
        obj_class_hat = tf.reshape(y_pred[:, S * S * 5:], [-1, S * S, C])

        # Convert the coordinates and width height into bottom-left, top-right coordinates

        obj_coord_conv = tf.concat(utils.coord_conversion(obj_coord, obj_hw * self.grid_size), axis=2)
        obj_coord_hat_conv = tf.concat(utils.coord_conversion(obj_coord_hat, obj_hw_hat * self.grid_size), axis=2)

        iou = utils.IOU(tf.concat([obj_coord_conv, obj_coord_hat_conv], axis=2))

        first_sum = tf.reduce_sum(
                        tf.multiply(obj_presence,
                                    tf.reduce_sum(
                                        tf.pow(obj_coord - obj_coord_hat, 2), axis=2)), axis=1)

        second_sum = tf.reduce_sum(
                        tf.multiply(obj_presence,
                                    tf.reduce_sum(
                                        tf.pow(obj_hw - obj_hw_hat, 2), axis=2)), axis=1)

        third_sum = tf.reduce_sum(
                        tf.multiply(obj_presence,
                                    tf.pow(obj_presence - iou * obj_presence_hat, 2)), keep_dims=True)

        fourth_sum = tf.reduce_sum(
                        tf.multiply(obj_not_presence,
                                    tf.pow(obj_presence - iou * obj_presence_hat, 2)), keep_dims=True)

        fifth_sum = tf.reduce_sum(
                        tf.multiply(obj_presence,
                                    tf.reduce_sum(
                                        tf.pow(obj_class - obj_class_hat, 2), axis=2)), axis=1)

        loss = self.lc * first_sum + \
               self.lc * second_sum + \
               third_sum + \
               self.noj * fourth_sum + \
               fifth_sum

        return loss

    def compile(self, optimizer, metrics=None):
        if metrics is not None:
            super().compile(optimizer=optimizer, loss=self.yolo_loss, metrics=metrics)
        else:
            super().compile(optimizer=optimizer, loss=self.yolo_loss)



#
# def yolo_loss(y_pred, y_true, S):
#
#     # y_true will be of the shape (S, S ,15), we are unwrapping
#     obj_presence = y_true[:S*S]
#     obj_not_presence = 1 - obj_presence
#     obj_coord = y_true[S*S:S*S*3].reshape(-1, 2)
#     obj_hw = y_true[S*S*3:S*S*5].reshape(-1, 2)
#     obj_class = y_true[S*S*5:].reshape(S*S, -1)
#
#
#     # Turn them into tf tensors
#     obj_presence = tf.Variable(obj_presence, dtype=tf.float32)
#     obj_not_presence = tf.Variable(obj_not_presence, dtype=tf.float32)
#     obj_coord = tf.Variable(obj_coord, dtype=tf.float32)
#     obj_hw = tf.exp(0.5 * tf.Variable(obj_coord, dtype=tf.float32))
#     obj_class = tf.Variable(obj_class, dtype=tf.float32)
#
#     # Do the same for y_pred
#     obj_presence_hat = tf.reshape(y_pred[:S*S], [-1])
#     obj_coord_hat = tf.reshape(y_pred[S*S:S*S*3], [-1, 2])
#     obj_hw_hat = tf.exp(0.5 * tf.reshape(y_pred[S*S*3:S*S*5], [-1, 2]))
#     obj_class_hat = tf.reshape(y_pred[S*S*5:], [S*S, -1])
#
#     # Convert the coordinates and width height into bottom-left, top-right coordinates
#     obj_coord_conv = tf.concat(center_width_height_to_bottom_left_top_right(obj_coord, obj_hw*28), axis=1)
#     obj_coord_hat_conv = tf.concat(center_width_height_to_bottom_left_top_right(obj_coord_hat, obj_hw_hat*28), axis=1)
#
#     # Calculate the intersection over union
#     iou = tf.squeeze(
#                 tf.map_fn(IOU,
#                           tf.concat([obj_coord_conv, obj_coord_hat_conv], axis=1)))
#
#
#
#     first_sum = tf.reduce_sum(
#         tf.multiply(obj_presence,
#                     tf.reduce_sum(
#                         tf.pow(obj_coord - obj_coord_hat, 2), axis=1)), keep_dims=True)
#
#     second_sum = tf.reduce_sum(
#         tf.multiply(obj_presence,
#                     tf.reduce_sum(
#                         tf.pow(obj_hw - obj_hw_hat, 2), axis=1)), keep_dims=True)
#
#     third_sum = tf.reduce_sum(
#         tf.multiply(obj_presence,
#                     tf.pow(obj_presence - iou * obj_presence_hat, 2)), keep_dims=True)
#
#     fourth_sum = tf.reduce_sum(
#         tf.multiply(obj_not_presence,
#                     tf.pow(obj_presence - iou * obj_presence_hat, 2)), keep_dims=True)
#
#     fifth_sum = tf.reduce_sum(
#         tf.multiply(obj_presence,
#                     tf.reduce_sum(
#                         tf.pow(obj_class - obj_class_hat, 2), axis=1)), keep_dims=True)
#
#     loss = first_sum + second_sum + third_sum +fourth_sum + fifth_sum
#     return loss
#
#     # for i in [loss, first_sum, second_sum, third_sum, fourth_sum, fifth_sum]:
#     #     print(K.ndim(i))
#     #
#     # with tf.Session() as sess:
#     #     sess.run(tf.global_variables_initializer())
#     #     print(sess.run(loss))
#
#
#
# yo = yolo((140, 140, 1),10,5 )
# yo.compile('adam')

