import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout

class yolo(Model):
    # Create an instance of the yolo model
    # Arguments:
    #   input_dimension: tuple, the dimensions of the input_image
    #   nb_classes: integer, indicates the number of classes YOLO should detect
    #   grid_size: integer, specifies the dividing grid size.
    def __init__(self,  input_dimension, nb_classes, grid_number):
        self.input_dim = input_dimension
        self.nb_classes = nb_classes
        self.grid_size = grid_number

        self.output_dimensions = grid_number * (1 + 4 + nb_classes)

        self.Input = Input(shape=input_dimension)
        self.Output = self._build_layers()

        super().__init__(inputs=self.Input, outputs=self.Output)

    def _build_layers(self):
        Input = self.Input

        x = Conv2D(filters=32, kernel_size=5, strides=1)(Input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1)(Input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=5, strides=1)(Input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=5, strides=1)(Input)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        x = Dense(self.output_dimensions)(x)

        return x


def yolo_loss(y_true):

    # y_true will be of the shape (S, S ,15)
    # we are unwrapping it to form a vector of length S^2 * 15

    obj_presence = y_true[:, :, 0].flatten()
    obj_coord = y_true[:, :, 1:3].flatten()
    obj_hw = y_true[:,:, 3:5].flatten()
    obj_class = y_true[:,:, 5:].flatten()

    # Define the masks in the yolo loss function


