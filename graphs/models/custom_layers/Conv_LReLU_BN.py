import tensorflow as tf
from tensorflow.python.keras.layers import Conv3D, LeakyReLU
from tensorflow.python.keras.initializers import RandomNormal


def _Conv3D_LReLU_BN(x_in, nf, kernel_size=3, strides=1, pad='valid', training=True, lrelu=True, batch_norm=True):
    init_var = RandomNormal(mean=0.0, stddev=0.02)
    x_out = Conv3D(nf, kernel_size=kernel_size, padding=pad, kernel_initializer=init_var, strides=strides)(x_in)
    if lrelu:
        x_out = LeakyReLU(0.2)(x_out)
    if batch_norm:
        x_out = tf.contrib.layers.batch_norm(x_out, decay=0.9, is_training=training, center=True, scale=True)
    return x_out
