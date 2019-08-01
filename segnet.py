import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications.mobilenet import MobileNet

from layers import BilinearUpsampling

from data_loader import INPUTSHAPE

def Upsampling4(init_weights=True, trainable=False, input=None):
    # Conform to functional API
    if input is None:
        return (lambda x: Upsampling4(init_weights=init_weights, trainable=trainable, input=x))

    def initializer(shape, dtype, **kwargs):
        grid = np.ogrid[:8, :8]
        k = (1 - abs(grid[0] - 3.5) / 4) * (1 - abs(grid[1] - 3.5) / 4)

        weights = np.zeros(shape, dtype=np.float32)
        for i in xrange(shape[-1]):
            weights[:, :, i, i] = k

        return K.backend.cast_to_floatx(weights)

    x = input
    if init_weights:
        x = Conv2DTranspose(int(input.shape[-1]), kernel_size=8, strides=4, padding="same", use_bias=False, kernel_initializer=initializer, trainable=trainable)(x)
    else:
        x = Conv2DTranspose(int(input.shape[-1]), kernel_size=8, strides=4, padding="same", use_bias=False)(x)

    return x

def SegNet(input_shape=(INPUTSHAPE, INPUTSHAPE, 3)):
    input = Input(shape=input_shape)

    mobilenet = MobileNet(input_tensor=input, input_shape=input_shape, alpha=0.5, include_top=False, weights="imagenet", pooling=None)

    x = mobilenet.get_layer("conv_pw_5_relu").output

    x = Conv2D(128, (1, 1), padding="same", use_bias=False, activation="relu")(x)
    x = Conv2D(64, (1, 1), padding="same", use_bias=False, activation="relu")(x)

    x = Conv2D(2, (1, 1), padding="same", use_bias=False)(x)

    x = BilinearUpsampling((8, 8))(x)

    x = Activation("softmax", name="decoder_softmax")(x)

    model = Model([input], x)

    return model
