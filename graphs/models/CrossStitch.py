import tensorflow as tf
from tensorflow.python.keras.layers import UpSampling3D, concatenate, Cropping3D
from graphs.models.custom_layers.Conv_LReLU_BN import _Conv3D_LReLU_BN


# Cross stitch unit on layers a and b. It has 2*2*num_filers trainable parameters.
#  Initialization of alphas: truncated random normal (mean=0.5, stddev=0.25).
def apply_cross_stitch(a, b, n_f):
    shape = tf.shape(a)
    newshape = [shape[0], shape[1]*shape[2]*shape[3], shape[4]]  # (bs, x, y, z, n_f) ==> (bs, x*y*z, n_f)

    a_flat = tf.reshape(a, newshape)  # [bs][x*y*z][n_f]
    b_flat = tf.reshape(b, newshape)  # [bs][x*y*z][n_f]

    a_flat = tf.transpose(a_flat, [0, 2, 1])  # [bs][n_f][x*y*z]
    b_flat = tf.transpose(b_flat, [0, 2, 1])  # [bs][n_f][x*y*z]

    a_flat = tf.expand_dims(a_flat, 2)  # [bs][n_f][1][x*y*z]
    b_flat = tf.expand_dims(b_flat, 2)  # [bs][n_f][1][x*y*z]
    a_concat_b = tf.concat(values=[a_flat, b_flat], axis=2)  # [bs][n_f][2][x*y*z]

    alphas = tf.get_variable(name="alphas", shape=(n_f, 2, 2), dtype=tf.float32,
                             collections=['cross_stitch_alphas', tf.GraphKeys.GLOBAL_VARIABLES],
                             initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.25), trainable=True)  # [n_f][2][2]
    alphas_tiled = tf.tile(tf.expand_dims(alphas, 0), [shape[0], 1, 1, 1])  # [bs][n_f][2][2]

    out = tf.matmul(alphas_tiled, a_concat_b)  # [bs][n_f][2][2] * [bs][n_f][2][x*y*z] ==> [bs][n_f][2][x*y*z]
    out = tf.transpose(out, [2, 0, 3, 1])  # [2][bs][x*y*z][n_f]

    out_a = out[0, :, :, :]  # [bs][x*y*z][n_f]
    out_b = out[1, :, :, :]  # [bs][x*y*z][n_f]

    out_a = tf.reshape(out_a, shape)  # [bs][x][y][z][n_f]
    out_b = tf.reshape(out_b, shape)  # [bs][x][y][z][n_f]

    return out_a, out_b


def joint_unet_crossStitchRAdam(fixed_image, moving_image, moving_segmentation, training, num_seg_classes=5, num_reg_classes=3,
                                name='joint_unet_crossStitchRAdam', reuse=False):

    input_image_reg = concatenate([fixed_image, moving_image], axis=-1)
    input_image_reg = concatenate([input_image_reg, tf.to_float(moving_segmentation)], axis=-1)  # Third channel with moving segmentation.

    enc_nf = [16, 32, 64]

    with tf.device('/device:GPU:0'):
        with tf.variable_scope(name, reuse=reuse):

            x0_seg = _Conv3D_LReLU_BN(fixed_image, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-2)^3, 16)
            x1_seg = _Conv3D_LReLU_BN(x0_seg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n-4)^3, 16)
            x2_seg = _Conv3D_LReLU_BN(x1_seg, enc_nf[0], kernel_size=3, strides=2, pad='same', training=training)     # (batch, (n/2-2)^3, 16)

            x0_reg = _Conv3D_LReLU_BN(input_image_reg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-2)^3, 16)
            x1_reg = _Conv3D_LReLU_BN(x0_reg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-4)^3, 16)
            x2_reg = _Conv3D_LReLU_BN(x1_reg, enc_nf[0], kernel_size=3, strides=2, pad='same', training=training)  # (batch, (n/2-2)^3, 16)

            with tf.variable_scope("cross_stitch_1"):
                stitched_x2_seg, stitched_x2_reg = apply_cross_stitch(x2_seg, x2_reg, enc_nf[0])

            x3_seg = _Conv3D_LReLU_BN(stitched_x2_seg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/2-4)^3, 32)
            x4_seg = _Conv3D_LReLU_BN(x3_seg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/2-6)^3, 32)
            x5_seg = _Conv3D_LReLU_BN(x4_seg, enc_nf[1], kernel_size=3, strides=2, pad='same', training=training)     # (batch, (n/4-3)^3, 32)

            x3_reg = _Conv3D_LReLU_BN(stitched_x2_reg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n/2-4)^3, 32)
            x4_reg = _Conv3D_LReLU_BN(x3_reg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n/2-6)^3, 32)
            x5_reg = _Conv3D_LReLU_BN(x4_reg, enc_nf[1], kernel_size=3, strides=2, pad='same', training=training)  # (batch, (n/4-3)^3, 32)

            with tf.variable_scope("cross_stitch_2"):
                stitched_x5_seg, stitched_x5_reg = apply_cross_stitch(x5_seg, x5_reg, enc_nf[1])

            x6_seg = _Conv3D_LReLU_BN(stitched_x5_seg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/4-5)^3, 64)
            x7_seg = _Conv3D_LReLU_BN(x6_seg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/4-7)^3, 64)
            x_low_res_seg = _Conv3D_LReLU_BN(x7_seg, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) #(batch, (n/4-7)^3, num_classes)

            x6_reg = _Conv3D_LReLU_BN(stitched_x5_reg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/4-5)^3, 64)
            x7_reg = _Conv3D_LReLU_BN(x6_reg, enc_nf[2], kernel_size=3, strides=1, pad='valid', training=training)    # (batch, (n/4-7)^3, 64)
            x_low_res_reg = _Conv3D_LReLU_BN(x7_reg, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) #(batch, (n/4-7)^3, num_classes)

            x9_seg = UpSampling3D(size=(2, 2, 2))(x7_seg)  # (batch, (n/2-14)^3, 64)
            x9_reg = UpSampling3D(size=(2, 2, 2))(x7_reg)  # (batch, (n/2-14)^3, 64)

            with tf.variable_scope("cross_stitch_3"):
                stitched_x9_seg, stitched_x9_reg = apply_cross_stitch(x9_seg, x9_reg, enc_nf[2])

            x4_cropped_seg = Cropping3D(4)(x4_seg)          # (batch, (n/2-14)^3, 32)
            stitched_x9_seg = concatenate([stitched_x9_seg, x4_cropped_seg])      # (batch, (n/2-14)^3, 96)
            x10_seg = _Conv3D_LReLU_BN(stitched_x9_seg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)   # (batch, (n/2-16)^3, 32)
            x11_seg = _Conv3D_LReLU_BN(x10_seg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n/2-18)^3, 32)
            x_mid_res_seg = _Conv3D_LReLU_BN(x11_seg, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training) # (batch, (n/2-18)^3, num_classes)

            x4_cropped_reg = Cropping3D(4)(x4_reg)  # (batch, (n/2-14)^3, 32)
            stitched_x9_reg = concatenate([stitched_x9_reg, x4_cropped_reg])  # (batch, (n/2-14)^3, 96)
            x10_reg = _Conv3D_LReLU_BN(stitched_x9_reg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n/2-16)^3, 32)
            x11_reg = _Conv3D_LReLU_BN(x10_reg, enc_nf[1], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n/2-18)^3, 32)
            x_mid_res_reg = _Conv3D_LReLU_BN(x11_reg, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False,
                                         training=training)  # (batch, (n/2-18)^3, num_classes)

            x13_seg = UpSampling3D(size=(2, 2, 2))(x11_seg)  # (batch, (n-36)^3, 32)
            x13_reg = UpSampling3D(size=(2, 2, 2))(x11_reg)  # (batch, (n-36)^3, 32)

            with tf.variable_scope("cross_stitch_4"):
                stitched_x13_seg, stitched_x13_reg = apply_cross_stitch(x13_seg, x13_reg, enc_nf[1])

            x1_cropped_seg = Cropping3D(16)(x1_seg)           # (batch, (n-36)^3, 16)
            stitched_x13_seg = concatenate([stitched_x13_seg, x1_cropped_seg])      # (batch, (n-36)^3, 48)
            x14_seg = _Conv3D_LReLU_BN(stitched_x13_seg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-38)^3, 16)
            x15_seg = _Conv3D_LReLU_BN(x14_seg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-40)^3, 16)
            x_high_res_seg = _Conv3D_LReLU_BN(x15_seg, num_seg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, (n-40)^3, num_classes)

            x1_cropped_reg = Cropping3D(16)(x1_reg)           # (batch, (n-36)^3, 16)
            stitched_x13_reg = concatenate([stitched_x13_reg, x1_cropped_reg])      # (batch, (n-36)^3, 48)
            x14_reg = _Conv3D_LReLU_BN(stitched_x13_reg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-38)^3, 16)
            x15_reg = _Conv3D_LReLU_BN(x14_reg, enc_nf[0], kernel_size=3, strides=1, pad='valid', training=training)  # (batch, (n-40)^3, 16)
            x_high_res_reg = _Conv3D_LReLU_BN(x15_reg, num_reg_classes, kernel_size=1, strides=1, pad='same', lrelu=False, batch_norm=False, training=training)  # (batch, (n-40)^3, num_classes)

            #  high:(n/4-7)^3;   mid:(n/2-18)^3;   low:(n-40)^3;
            return x_high_res_seg, x_mid_res_seg, x_low_res_seg, x_high_res_reg, x_mid_res_reg, x_low_res_reg
