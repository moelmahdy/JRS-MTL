import tensorflow as tf
from tensorflow.python.keras.layers import UpSampling3D, concatenate, Cropping3D
from graphs.models.custom_layers.Conv_LReLU_BN import _Conv3D_LReLU_BN


def SEDD(image, training, num_seg_classes=5, num_reg_classes=3, name='SEDD', reuse=False):

    enc_nf = [16, 32, 64]

    with tf.device('/device:GPU:0'):
        with tf.variable_scope(name, reuse=reuse):
            x_in = image  # (batch, 64, 64, 64, 1)
            x0 = _Conv3D_LReLU_BN(x_in, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 62, 62, 62, 16)
            x1 = _Conv3D_LReLU_BN(x0, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, 60, 60, 60, 16)
            x2 = _Conv3D_LReLU_BN(x1, enc_nf[0], kernel_size=3, strides=2, pad='same', training=training)     # (batch, 30, 30, 30, 16)
            x3 = _Conv3D_LReLU_BN(x2, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, 28, 28, 28, 32)
            x4 = _Conv3D_LReLU_BN(x3, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, 26, 26, 26, 32)
            ###########################
            x5_seg = _Conv3D_LReLU_BN(x4, enc_nf[1], kernel_size=3, strides=2, pad='same', training=training)     # (batch, 13, 13, 13, 32)
            x6_seg = _Conv3D_LReLU_BN(x5_seg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, 11, 11, 11, 64)
            x7_seg = _Conv3D_LReLU_BN(x6_seg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, 9, 9, 9, 64)
            x_low_res_seg = _Conv3D_LReLU_BN(x7_seg, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) #(batch, 9, 9, 9, num_classes)

            x9_seg = UpSampling3D(size=(2, 2, 2))(x7_seg)  # (batch, 18, 18, 18, 64)
            x4_cropped = Cropping3D(4)(x4)          # (batch, 18, 18, 18, 32)
            x9_seg = concatenate([x9_seg, x4_cropped])      # (batch, 18, 18, 18, 96)
            x10_seg = _Conv3D_LReLU_BN(x9_seg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)   # (batch, 16, 16, 16, 32)
            x11_seg = _Conv3D_LReLU_BN(x10_seg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 14, 14, 14, 32)
            x_mid_res_seg = _Conv3D_LReLU_BN(x11_seg, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) # (batch, 14, 14, 14, num_classes)

            x13_seg = UpSampling3D(size=(2, 2, 2))(x11_seg)  # (batch, 28, 28, 28, 32)
            x1_cropped = Cropping3D(16)(x1)           # (batch, 28, 28, 28, 16)
            x13_seg = concatenate([x13_seg, x1_cropped])      # (batch, 28, 28, 28, 48)
            x14_seg = _Conv3D_LReLU_BN(x13_seg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 26, 26, 26, 16)
            x15_seg = _Conv3D_LReLU_BN(x14_seg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 24, 24, 24, 16)
            x_high_res_seg = _Conv3D_LReLU_BN(x15_seg, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, 24, 24, 24, num_classes)

            ###########################
            x5_reg = _Conv3D_LReLU_BN(x4, enc_nf[1], kernel_size=3, strides=2, pad='same', training=training)  # (batch, 13, 13, 13, 32)
            x6_reg = _Conv3D_LReLU_BN(x5_reg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 11, 11, 11, 64)
            x7_reg = _Conv3D_LReLU_BN(x6_reg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 9, 9, 9, 64)
            x_low_res_reg = _Conv3D_LReLU_BN(x7_reg, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False,
                                        batch_norm=False, training=training)  # (batch, 9, 9, 9, num_classes)

            x9_reg = UpSampling3D(size=(2, 2, 2))(x7_reg)  # (batch, 18, 18, 18, 64)
            #x4_cropped
            x9_reg = concatenate([x9_reg, x4_cropped])  # (batch, 18, 18, 18, 96)
            x10_reg = _Conv3D_LReLU_BN(x9_reg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 16, 16, 16, 32)
            x11_reg = _Conv3D_LReLU_BN(x10_reg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 14, 14, 14, 32)
            x_mid_res_reg = _Conv3D_LReLU_BN(x11_reg, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, 14, 14, 14, num_classes)

            x13_reg = UpSampling3D(size=(2, 2, 2))(x11_reg)  # (batch, 28, 28, 28, 32)
            #x1_cropped
            x13_reg = concatenate([x13_reg, x1_cropped])  # (batch, 28, 28, 28, 48)
            x14_reg = _Conv3D_LReLU_BN(x13_reg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 26, 26, 26, 16)
            x15_reg = _Conv3D_LReLU_BN(x14_reg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, 24, 24, 24, 16)
            x_high_res_reg = _Conv3D_LReLU_BN(x15_reg, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, 24, 24, 24, num_classes)

            return x_high_res_seg, x_mid_res_seg, x_low_res_seg, x_high_res_reg, x_mid_res_reg, x_low_res_reg