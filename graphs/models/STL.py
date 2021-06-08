import tensorflow as tf
from tensorflow.python.keras.layers import UpSampling3D, concatenate, Cropping3D
from graphs.models.custom_layers.Conv_LReLU_BN import _Conv3D_LReLU_BN

def STL(fixed_image, moving_image, moving_segmentation, training, isSeg, num_seg_classes=5,
        num_reg_classes=3, name='STL', reuse=False, addSMToReg=True, addSMAndIMToSeg=False,
        addSMToSeg=False, addIMToSeg=False):


    enc_nf = [23, 45, 91]

    with tf.device('/device:GPU:0'):
        with tf.variable_scope(name, reuse=reuse):

            if isSeg:
                x_in = fixed_image  # (batch, (n)^3, 1)
                if addSMAndIMToSeg:
                    x_in = concatenate([x_in, moving_image], axis=-1)
                    x_in = concatenate([x_in, tf.to_float(moving_segmentation)], axis=-1)  # Third channel with moving segmentation.
                if addIMToSeg:
                    x_in = concatenate([x_in, moving_image], axis=-1)
                if addSMToSeg:
                    x_in = concatenate([x_in, tf.to_float(moving_segmentation)], axis=-1)
            else:
                x_in = concatenate([fixed_image, moving_image], axis=-1)
                if addSMToReg:
                    x_in = concatenate([x_in, tf.to_float(moving_segmentation)], axis=-1)  # Third channel with moving segmentation.
            x0 = _Conv3D_LReLU_BN(x_in, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-2)^3, 16)
            x1 = _Conv3D_LReLU_BN(x0, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n-4)^3, 16)
            x2 = _Conv3D_LReLU_BN(x1, enc_nf[0], kernel_size=3, strides=2, pad='same', training=training)     # (batch, (n/2-2)^3, 16)
            x3 = _Conv3D_LReLU_BN(x2, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/2-4)^3, 32)
            x4 = _Conv3D_LReLU_BN(x3, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/2-6)^3, 32)
            x5 = _Conv3D_LReLU_BN(x4, enc_nf[1], kernel_size=3, strides=2, pad='same', training=training)     # (batch, (n/4-3)^3, 32)
            x6 = _Conv3D_LReLU_BN(x5, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/4-5)^3, 64)
            x7 = _Conv3D_LReLU_BN(x6, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/4-7)^3, 64)

            x9 = UpSampling3D(size=(2, 2, 2))(x7)  # (batch, (n/2-14)^3, 64)
            x4_cropped = Cropping3D(4)(x4)          # (batch, (n/2-14)^3, 32)
            x9 = concatenate([x9, x4_cropped])      # (batch, (n/2-14)^3, 96)
            x10 = _Conv3D_LReLU_BN(x9, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)   # (batch, (n/2-16)^3, 32)
            x11 = _Conv3D_LReLU_BN(x10, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n/2-18)^3, 32)

            x13 = UpSampling3D(size=(2, 2, 2))(x11)  # (batch, (n-36)^3, 32)
            x1_cropped = Cropping3D(16)(x1)           # (batch, (n-36)^3, 16)
            x13 = concatenate([x13, x1_cropped])      # (batch, (n-36)^3, 48)
            x14 = _Conv3D_LReLU_BN(x13, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-38)^3, 16)
            x15 = _Conv3D_LReLU_BN(x14, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-40)^3, 16)

            if isSeg:
                x_low_res_seg = _Conv3D_LReLU_BN(x7, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) #(batch, (n/4-7)^3, num_classes)
                x_mid_res_seg = _Conv3D_LReLU_BN(x11, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) # (batch, (n/2-18)^3, num_classes)
                x_high_res_seg = _Conv3D_LReLU_BN(x15, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, (n-40)^3, num_classes)
                shape_high = tf.shape(x_high_res_seg)
                x_high_res_reg_dims = tf.stack([shape_high[0], shape_high[1], shape_high[2], shape_high[3], num_reg_classes])
                x_high_res_reg = tf.fill(x_high_res_reg_dims, 0.0)
                shape_mid = tf.shape(x_mid_res_seg)
                x_mid_res_reg_dims = tf.stack([shape_mid[0], shape_mid[1], shape_mid[2], shape_mid[3], num_reg_classes])
                x_mid_res_reg = tf.fill(x_mid_res_reg_dims, 0.0)
                shape_low = tf.shape(x_low_res_seg)
                x_low_res_reg_dims = tf.stack([shape_low[0], shape_low[1], shape_low[2], shape_low[3], num_reg_classes])
                x_low_res_reg = tf.fill(x_low_res_reg_dims, 0.0)
            else: #isReg
                x_low_res_reg = _Conv3D_LReLU_BN(x7, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) #(batch, (n/4-7)^3, num_classes)
                x_mid_res_reg = _Conv3D_LReLU_BN(x11, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) # (batch, (n/2-18)^3, num_classes)
                x_high_res_reg = _Conv3D_LReLU_BN(x15, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, (n-40)^3, num_classes)
                shape_high = tf.shape(x_high_res_reg)
                x_high_res_seg_dims = tf.stack([shape_high[0], shape_high[1], shape_high[2], shape_high[3], num_seg_classes])
                x_high_res_seg = tf.fill(x_high_res_seg_dims, 0.0)
                shape_mid = tf.shape(x_mid_res_reg)
                x_mid_res_seg_dims = tf.stack([shape_mid[0], shape_mid[1], shape_mid[2], shape_mid[3], num_seg_classes])
                x_mid_res_seg = tf.fill(x_mid_res_seg_dims, 0.0)
                shape_low = tf.shape(x_low_res_reg)
                x_low_res_seg_dims = tf.stack([shape_low[0], shape_low[1], shape_low[2], shape_low[3], num_seg_classes])
                x_low_res_seg = tf.fill(x_low_res_seg_dims, 0.0)

            return x_high_res_seg, x_mid_res_seg, x_low_res_seg, x_high_res_reg, x_mid_res_reg, x_low_res_reg