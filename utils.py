import numpy as np
import tensorflow as tf


def unwrap_data(data):
    # helper function to unwrap the confidence, coordinates and classes
    # generate from the mnist_documents class

    # Arguments:
        # data: array, with shape (S, S, 5+C)
    # return:
        # array: Array consisting of confidence, coordinates , width_height ,classes.

    confidence = data[:, :, 0].flatten()
    coordinates, width_height = data[:, :, 1:3].flatten(), data[:, :, 3:5].flatten()
    classes = data[:, :, 5:].flatten()

    return np.r_[confidence, coordinates, width_height, classes]

def coord_conversion(coordinates, width_height):
    # turn the center coordinates, width and height of a bounding box
    # to bottom-left and top-right coordinates
    # Arguments:
    #   coordinates: array, with shape (None, 2)
    #   width_height: array, with_shape (None, 2)

    width_height /= 2

    bottom_left = coordinates - width_height
    top_right = coordinates + width_height

    return bottom_left, top_right

def IOU(bounding_boxes):
    # helper function to help calculate the intersection over union when
    # given two bounding boxes.
    # Arguments:
    #   bounding_boxes: array, with shape (None, None, 8)

    x = bounding_boxes[:, :, ::2]
    y = bounding_boxes[:, :, 1::2]

    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    y1 = y[:, :, ::2]
    y2 = y[:, :, 1::2]

    x11 = tf.reduce_max(x1, axis=2)
    x12 = tf.reduce_min(x1, axis=2)

    x21 = tf.reduce_max(x2, axis=2)
    x22 = tf.reduce_min(x2, axis=2)

    y11 = tf.reduce_max(y1, axis=2)
    y12 = tf.reduce_min(y1, axis=2)

    y21 = tf.reduce_max(y2, axis=2)
    y22 = tf.reduce_min(y2, axis=2)

    inter_area = (x11 - x22 + 1) * (y12 - y21 + 1)

    box1_area = (x2[:, :, 0] - x1[: ,: ,0] + 1) * (y1[:, :, 0] - y2[:, :, 0] + 1)
    box2_area = (x2[:, :, 1] - x1[: ,: ,1] + 1) * (y1[:, :, 1] - y2[:, :, 1] + 1)

    return inter_area / (box1_area + box2_area - inter_area)

    # x11, y11, x12, y12, x21, y21, x22, y22 = tf.split(bounding_boxes, 8)
    #
    # # print(tf.split(bounding_boxes, 8))
    #
    # xI1 = tf.maximum(x11, tf.transpose(x21))
    # yI1 = tf.maximum(y11, tf.transpose(y21))
    #
    # xI2 = tf.minimum(x12, tf.transpose(x22))
    # yI2 = tf.minimum(y12, tf.transpose(y22))
    #
    # inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)
    #
    # bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    # bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
    #
    # return inter_area / ((bboxes1_area + tf.transpose(bboxes2_area)) - inter_area)

# def get_training_data(sample_nb):
