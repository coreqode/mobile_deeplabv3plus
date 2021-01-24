import re
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import (
    Layer,
    Input,
    SeparableConv2D,
    SeparableConv2D,
    ZeroPadding2D,
    ReLU,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras import backend as B
from mobilenetv2 import mobilenetv2


def decoder(x, low_level, num_classes):
    low_level = SeparableConv2D(48, 1)(low_level)
    low_level = BatchNormalization()(low_level)
    low_level = ReLU()(low_level)
    x = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(x)
    x = tf.keras.layers.concatenate([x, low_level], axis=-1)
    x = SeparableConv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = SeparableConv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = SeparableConv2D(num_classes, 1)(x)
    return x


def aspp_module(x, planes, kernel_size, padding, dilation):
    x = ZeroPadding2D(padding)(x)
    x = SeparableConv2D(planes, kernel_size, 1, dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def aspp(x, output_stride):
    if output_stride == 16:
        dilations = [1, 6, 12, 18]
    elif output_stride == 8:
        dilations = [1, 12, 24, 36]
    else:
        raise NotImplementedError
    x1 = aspp_module(x, 256, 1, padding=0, dilation=dilations[0])
    x2 = aspp_module(x, 256, 3, padding=dilations[1], dilation=dilations[1])
    x3 = aspp_module(x, 256, 3, padding=dilations[2], dilation=dilations[2])
    x4 = aspp_module(x, 256, 3, padding=dilations[3], dilation=dilations[3])
    x5 = GlobalAveragePooling2D()(x)
    x5 = tf.keras.layers.Reshape((1, 1, 320))(x5)
    x5 = SeparableConv2D(256, 1)(x5)
    x5 = BatchNormalization()(x5)
    x5 = ReLU()(x5)
    x5 = tf.keras.layers.UpSampling2D(
        size=(x4.shape[1], x4.shape[2]), interpolation="bilinear"
    )(x5)
    x = tf.keras.layers.concatenate([x1, x2, x3, x4, x5], axis=-1)
    x = SeparableConv2D(256, 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    return x


def deeplabv3(inputs, input_shape, num_classes, output_stride, backbone="mobilenet"):
    if backbone == "mobilenet":
        lowlevel, x = mobilenetv2(inputs)
    else:
        raise ValueError(f"{backbone} is not integrated yet.")
    x = aspp(x, output_stride)
    x = decoder(x, lowlevel, num_classes)
    h, w = inputs.shape[1] // x.shape[1], inputs.shape[2] // x.shape[2]
    x = tf.keras.layers.UpSampling2D(size=(h, w), interpolation="bilinear")(x)
    return x


if __name__ == "__main__":
    inputs = Input([256, 192, 3])
    out = deeplabv3(inputs, (256, 192, 3), 15, 8)
    model = Model(inputs, out)
    print(model.summary())
