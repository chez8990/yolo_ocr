import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers imoprt Input, Dense, Conv2D, MaxPooling2D, Dropout

class yolo(Model):
    # Create an instance of the yolo model
    # Arguments:
    #   input_dimension: tuple, the dimensions of the input_image
    #   nb_classes: integer, indicates the number of classes YOLO should detect
    #   grid_size: integer, specifies the dividing grid size.
    def __init__(self,  input_dimension, nb_classes, grid_size):
        self.input_dim = input_dimension
        self.nb_classes = nb_classes
        self.grid_size = grid_size

        self.model = _build_layers()
        super().__init__()

    def _build_layers(self):
        Input = Input(shape=self.input_dimension)

        x = Conv2D(filters=32, kernel_size=5, strides=1)(Input)
        x = MaxPooling(pool_size=(2, 2))(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1)(Input)
        x = MaxPooling(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=5, strides=1)(Input)
        x = MaxPooling(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=5, strides=1)(Input)
        x = MaxPooling(pool_size=(2, 2))(x)

        x = Dense(1028, activation='sigmoid')