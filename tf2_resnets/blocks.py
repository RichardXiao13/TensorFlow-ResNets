import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tf2_resnets import convolutions
layers.Conv2D = convolutions.Conv2D
from tf2_resnets.split_attention import SplitAttentionConv2D


def block1(x,
          filters,
          kernel_size=3,
          stride=1,
          groups=1,
          base_width=64,
          conv_shortcut=True,
          name=None):
  """A residual block.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    groups: default 1, group size for grouped convolution.
    base_width: default 64, filters of the intermediate layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  width = int(filters * (base_width / 64.)) * groups
  expansion = 4

  if conv_shortcut:
    shortcut = layers.Conv2D(
        filters * expansion,
        1,
        strides=stride,
        use_bias=False,
        name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(width, 1, use_bias=False, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(
      width,
      kernel_size=kernel_size,
      strides=stride,
      use_bias=False,
      groups=groups,
      name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(
      filters * expansion, 1, use_bias=False, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x


def stack1(x, filters, blocks, stride1=2, groups=1, base_width=64, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    groups: default 1, group size for grouped convolution.
    base_width: default 64, filters of the intermediate layer in a block.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block1(x, filters, stride=stride1, groups=groups, base_width=base_width, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block1(
        x,
        filters,
        groups=groups,
        conv_shortcut=False,
        base_width=base_width,
        name=name + '_block' + str(i))
  return x


def block2(x,
          filters,
          kernel_size=3,
          stride=1,
          groups=1,
          base_width=64,
          radix=1,
          is_first=True,
          conv_shortcut=True,
          name=None):
  """A residual block.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    groups: default 1, group size for grouped convolution.
    base_width: default 64, filters of the intermediate layer.
    radix: default 1, splits for split attention layer.
    is_first: default True, when to use pooling layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  width = int(filters * (base_width / 64.)) * groups
  expansion = 4
  old_stride = stride

  if stride > 1 or is_first:
    stride = 1

  if conv_shortcut:
    shortcut = layers.AveragePooling2D(
      pool_size=stride, strides=old_stride, name=name + '_0_pool')(x)
    shortcut = layers.Conv2D(
        filters * expansion,
        1,
        strides=1,
        use_bias=False,
        name=name + '_0_conv')(shortcut)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(width, 1, use_bias=False, name=name + '_1_conv')(x)

  x = layers.BatchNormalization(
    axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = SplitAttentionConv2D(
      x,
      width,
      kernel_size=kernel_size,
      stride=stride,
      padding=(1, 1),
      groups=groups,
      radix=radix,
      use_bias=False,
      name=name + '_2_conv')

  if stride > 1 or is_first:
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_0_pad')(x)
    x = layers.AveragePooling2D(
      pool_size=kernel_size, strides=old_stride, name=name + '_1_pool')(x)

  x = layers.Conv2D(
      filters * expansion, 1, use_bias=False, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x


def stack2(x, filters, blocks, stride1=2, groups=1, base_width=64, radix=1, is_first=True, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    groups: default 32, group size for grouped convolution.
    base_width: default 64, filters of the intermediate layer in a block.
    radix: default 1, splits for split attention layer in a block.
    is_first: default True, when to use pooling layer in a block.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block2(x,
            filters,
            stride=stride1,
            groups=groups,
            base_width=base_width,
            radix=radix,
            is_first=is_first,
            name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block2(
        x,
        filters,
        groups=groups,
        conv_shortcut=False,
        base_width=base_width,
        radix=radix,
        is_first=is_first,
        name=name + '_block' + str(i))
  return x
