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


def yolo_loss(y_pred, y_true):
