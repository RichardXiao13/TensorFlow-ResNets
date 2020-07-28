import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tf2_resnets import convolutions
layers.Conv2D = convolutions.Conv2D


def RSoftmax(x, filters, radix, groups, name):
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  c = filters // radix // groups
  shape = (groups, radix, c) if bn_axis == 3 else (groups, radix, c)

  x = layers.Reshape(shape, name=name + '_0_attn_reshape')(x)
  x = layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)), name=name + '_attn_transpose')(x)
  x = layers.Softmax(axis=1, name=name + '_attn_softmax')(x)

  shape = (1, 1, filters) if bn_axis == 3 else (filters, 1, 1)
  x = layers.Reshape(shape, name=name + '_1_attn_reshape')(x)
  return x


def SplitAttentionConv2D(x,
                         filters,
                         kernel_size,
                         stride=1,
                         padding=(0, 0),
                         groups=1,
                         use_bias=True,
                         radix=2,
                         name=None):
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  reduction_factor = 4
  inter_filters = max(filters * radix // reduction_factor, 32)
  x = layers.ZeroPadding2D((padding, padding), name=name + '_splat_pad')(x)
  x = layers.Conv2D(
      filters * radix,
      kernel_size=kernel_size,
      strides=stride,
      groups=groups * radix,
      use_bias=use_bias,
      name=name + '_0_splat_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_splat_bn')(x)
  x = layers.Activation('relu', name=name + '_0_splat_relu')(x)

  splits = layers.Lambda(
      lambda x: tf.split(x, radix, bn_axis), name=name + '_0_splat_split')(x)
  x = layers.Add(name=name + '_0_splat_add')(splits)
  x = layers.GlobalAveragePooling2D(name=name + '_0_splat_pool')(x)
  shape = (1, 1, filters) if bn_axis == 3 else (filters, 1, 1)
  x = layers.Reshape(shape, name=name + '_0_splat_reshape')(x)

  x = layers.Conv2D(
      inter_filters, kernel_size=1, groups=groups, name=name + '_1_splat_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_splat_bn')(x)
  x = layers.Activation('relu', name=name + '_1_splat_relu')(x)

  # Attention
  x = layers.Conv2D(
      filters * radix, kernel_size=1, groups=groups, name=name + '_2_splat_conv')(x)
  x = RSoftmax(x, filters * radix, radix, groups, name=name)
  x = layers.Lambda(
      lambda x: tf.split(x, radix, bn_axis), name=name + '_1_splat_split')(x)
  x = layers.Lambda(
      lambda x: [tf.stack(x[0], axis=bn_axis), tf.stack(x[1], axis=bn_axis)], name=name + '_splat_stack')([splits, x])
  x = layers.Multiply(name=name + '_splat_mult')(x)
  x = layers.Lambda(
      lambda x: tf.unstack(x, axis=bn_axis), name=name + '_splat_unstack')(x)
  x = layers.Add(name=name + '_splat_add')(x)
  return x
