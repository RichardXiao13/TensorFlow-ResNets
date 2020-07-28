"""ResNet models for Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tf2_resnets import convolutions
layers.Conv2D = convolutions.Conv2D
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils


BASE_WEIGHTS_PATH = (
    'https://github.com/RichardXiao13/Tensorflow-ResNet/releases/download/'
)
IMAGENET_WEIGHTS_HASHES = {
    'resnext50': ('ae0d1246ce78fc84e96b68815980d252',
                  '640d25b086d52a95d1516669eac5cc99'),
    'resnext101': ('5c0275334d267f617c1fa4c87c49f36f',
                   '2415217b162385a087f13867b0162dfe'),
    'wide_resnet50': ('a73fd180ce246e8771cbab225ff03f07',
                      '339ac967cce04f0eb0756f60b80fd79b'),
    'wide_resnet101': ('f0d3a55b905bbcf69a94001b7e7f65cd',
                       '5640917173c841166272114b0259f8e7'),
    'resnest50': ('ecb1254d3d33cd767497bf061b63b064',
                  'fa4b3f14e6d287225ddcd70803ee2faa'),
    'resnest101': ('d87c42c5f3edca88a3f83057cffd96ca',
                   'f070dc0d1863a2b31f4094dd88713d3d'),
    'resnest200': ('515b4f4c86ef0923b65e84cc2e519845',
                   '83b49aa2528cecfbe2eb713077d05033'),
    'resnest269': ('efeade22e9992057d69c6727fe56eb77',
                   'c29b0d7e5cc68ebe37cbdfba17f3359b')
}
SSL_WEIGHTS_HASHES = {
    'resnext50': ('b216ae004757827896210dcc5b823298',
                  '71f76fb29ed62cb19a8c3788fcde72a7'),
    'resnext101': ('dbb2de52a36abea91a470a98c35c0874',
                   '040d7708388f60ffc63194d91501dcde')
}
SWSL_WEIGHTS_HASHES = {
    'resnext50': ('8f3231cea92036f748eb5391c2c1266c',
                  'abf020c713a5d5b2555a5169faf02146'),
    'resnext101': ('3255ad088217aede56d1b1f41f90eb3b',
                   '4813d8155b7c4bfa2228dffd745f4ad2')
}


def ResNet(stack_fn,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           deep_stem=False,
           stem_width=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
  """Instantiates the ResNeSt, ResNeXt, and Wide ResNet architecture.

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.resnet.preprocess_input` for an example.

  Arguments:
    stack_fn: a function that returns output tensor for the
      stacked residual blocks.
    use_bias: whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
    model_name: string, model name.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `channels_last` data format)
      or `(3, 224, 224)` (with `channels_first` data format).
      It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional layer.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    **kwargs: For backwards compatibility only.
  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  if 'layers' in kwargs:
    global layers
    layers = kwargs.pop('layers')
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))
  if not (weights in {'imagenet', 'ssl', 'swsl', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), `ssl` '
                     '(semi-supervised), `swsl` '
                     '(semi-weakly supervised), '
                     'or the path to the weights file to be loaded.')

  if (weights == 'imagenet' or weights == 'ssl' or weights == 'swsl') and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"`, '
                     'or `weights` as `"ssl"`, '
                     'or `weights` as `"swsl"`, '
                     'with `include_top` '
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if deep_stem:
    # Deep stem based off of ResNet-D
    x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name='conv1_0_pad')(img_input)
    x = layers.Conv2D(stem_width, 3, strides=2, use_bias=use_bias, name='conv1_0_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1_0_bn')(x)
    x = layers.Activation('relu', name='conv1_0_relu')(x)

    x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name='conv1_1_pad')(x)
    x = layers.Conv2D(stem_width, 3, strides=1, use_bias=use_bias, name='conv1_1_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1_1_bn')(x)
    x = layers.Activation('relu', name='conv1_1_relu')(x)

    x = layers.ZeroPadding2D(
      padding=((1, 1), (1, 1)), name='conv1_2_pad')(x)
    x = layers.Conv2D(stem_width * 2, 3, strides=1, use_bias=use_bias, name='conv1_2_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1_2_bn')(x)
    x = layers.Activation('relu', name='conv1_2_relu')(x)
  else:
    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
  x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

  x = stack_fn(x)

  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name=model_name)

  # Load weights.
  global BASE_WEIGHTS_PATH
  if 'resnest' in model_name:
    BASE_WEIGHTS_PATH += 'v0.2.0/'
  else:
    BASE_WEIGHTS_PATH += 'v0.1.0/'

  if (weights == 'imagenet') and (model_name in IMAGENET_WEIGHTS_HASHES):
    if include_top:
      file_name = model_name + '_imagenet_top.h5'
      file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
    else:
      file_name = model_name + '_imagenet_notop.h5'
      file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)

  elif (weights == 'ssl') and (model_name in SSL_WEIGHTS_HASHES):
    if include_top:
      file_name = model_name + '_ssl_top.h5'
      file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
    else:
      file_name = model_name + '_ssl_notop.h5'
      file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)

  elif (weights == 'swsl') and (model_name in SWSL_WEIGHTS_HASHES):
    if include_top:
      file_name = model_name + '_swsl_top.h5'
      file_hash = SWSL_WEIGHTS_HASHES[model_name][0]
    else:
      file_name = model_name + '_swsl_notop.h5'
      file_hash = SWSL_WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def preprocess_input(x, data_format=None):
  return imagenet_utils.preprocess_input(
      x, data_format=data_format, mode='torch')

preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='',
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH)
