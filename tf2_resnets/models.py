from tf2_resnets.resnet import ResNet
from tf2_resnets.blocks import stack1, stack2, stack3


def ResNeXt50(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
  """Instantiates the ResNeXt50 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, groups=32, base_width=4, name='conv2')
    x = stack1(x, 128, 4, groups=32, base_width=4, name='conv3')
    x = stack1(x, 256, 6, groups=32, base_width=4, name='conv4')
    return stack1(x, 512, 3, groups=32, base_width=4, name='conv5')

  return ResNet(stack_fn, False, 'resnext50', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def ResNeXt101(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               **kwargs):
  """Instantiates the ResNeXt101 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, groups=32, base_width=8, name='conv2')
    x = stack1(x, 128, 4, groups=32, base_width=8, name='conv3')
    x = stack1(x, 256, 23, groups=32, base_width=8, name='conv4')
    return stack1(x, 512, 3, groups=32, base_width=8, name='conv5')

  return ResNet(stack_fn, False, 'resnext101', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def WideResNet50(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
  """Instantiates the Wide-ResNet50-2 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, base_width=128, name='conv2')
    x = stack1(x, 128, 4, base_width=128, name='conv3')
    x = stack1(x, 256, 6, base_width=128, name='conv4')
    return stack1(x, 512, 3, base_width=128, name='conv5')

  return ResNet(stack_fn, False, 'wide_resnet50', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def WideResNet101(include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  classes=1000,
                  **kwargs):
  """Instantiates the Wide-ResNet101-2 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, base_width=128, name='conv2')
    x = stack1(x, 128, 4, base_width=128, name='conv3')
    x = stack1(x, 256, 23, base_width=128, name='conv4')
    return stack1(x, 512, 3, base_width=128, name='conv5')

  return ResNet(stack_fn, False, 'wide_resnet101', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def ResNeSt50(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
  """Instantiates the ResNeSt50 architecture."""

  def stack_fn(x):
    x = stack2(x, 64, 3, stride1=1, base_width=64, radix=2, is_first=False, name='conv2')
    x = stack2(x, 128, 4, base_width=64, radix=2, name='conv3')
    x = stack2(x, 256, 6, base_width=64, radix=2, name='conv4')
    return stack2(x, 512, 3, base_width=64, radix=2, name='conv5')

  return ResNet(stack_fn, False, 'resnest50', include_top, weights,
                input_tensor, input_shape, pooling, True, 32, classes, **kwargs)


def ResNeSt101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
  """Instantiates the ResNeSt101 architecture."""

  def stack_fn(x):
    x = stack2(x, 64, 3, stride1=1, base_width=64, radix=2, is_first=False, name='conv2')
    x = stack2(x, 128, 4, base_width=64, radix=2, name='conv3')
    x = stack2(x, 256, 23, base_width=64, radix=2, name='conv4')
    return stack2(x, 512, 3, base_width=64, radix=2, name='conv5')

  return ResNet(stack_fn, False, 'resnest101', include_top, weights,
                input_tensor, input_shape, pooling, True, 64, classes, **kwargs)


def ResNeSt200(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
  """Instantiates the ResNeSt200 architecture."""

  def stack_fn(x):
    x = stack2(x, 64, 3, stride1=1, base_width=64, radix=2, is_first=False, name='conv2')
    x = stack2(x, 128, 24, base_width=64, radix=2, name='conv3')
    x = stack2(x, 256, 36, base_width=64, radix=2, name='conv4')
    return stack2(x, 512, 3, base_width=64, radix=2, name='conv5')

  return ResNet(stack_fn, False, 'resnest200', include_top, weights,
                input_tensor, input_shape, pooling, True, 64, classes, **kwargs)


def ResNeSt269(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
  """Instantiates the ResNeSt269 architecture."""

  def stack_fn(x):
    x = stack2(x, 64, 3, stride1=1, base_width=64, radix=2, is_first=False, name='conv2')
    x = stack2(x, 128, 30, base_width=64, radix=2, name='conv3')
    x = stack2(x, 256, 48, base_width=64, radix=2, name='conv4')
    return stack2(x, 512, 8, base_width=64, radix=2, name='conv5')

  return ResNet(stack_fn, False, 'resnest269', include_top, weights,
                input_tensor, input_shape, pooling, True, 64, classes, **kwargs)


def ResNet18(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            **kwargs):
  """Instantiates the ResNet18 architecture."""

  def stack_fn(x):
    x = stack3(x, 64, 2, stride1=1, conv_shortcut=False, name='conv2')
    x = stack3(x, 128, 2, name='conv3')
    x = stack3(x, 256, 2, name='conv4')
    return stack3(x, 512, 2, name='conv5')

  return ResNet(stack_fn, False, 'resnet18', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def ResNet34(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            **kwargs):
  """Instantiates the ResNet34 architecture."""

  def stack_fn(x):
    x = stack3(x, 64, 3, stride1=1, conv_shortcut=False, name='conv2')
    x = stack3(x, 128, 4, name='conv3')
    x = stack3(x, 256, 6, name='conv4')
    return stack3(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, 'resnet34', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def ResNet50(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            **kwargs):
  """Instantiates the ResNet50 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, 'resnet50', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def ResNet101(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            **kwargs):
  """Instantiates the ResNet101 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 23, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, 'resnet50', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


def ResNet152(include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            **kwargs):
  """Instantiates the ResNet152 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 8, name='conv3')
    x = stack1(x, 256, 36, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, 'resnet50', include_top, weights,
                input_tensor, input_shape, pooling, False, None, classes, **kwargs)


DOC = """


  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.
  
  Arguments:
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      'ssl' (semi-supervised),
      'swsl' (semi-weakly supervised),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
  Returns:
    A Keras model instance.
"""

WRSNTREF = """
    Reference:
      - [Deep Residual Learning for Image Recognition](
          https://arxiv.org/abs/1605.07146) (CVPR 2016)
"""

RSNXTREF = """
    Reference:
      - [Deep Residual Learning for Image Recognition](
          https://arxiv.org/abs/1611.05431) (CVPR 2016)
"""

RSNSTREF = """
    Reference:
      - [Deep Residual Learning for Image Recognition](
          https://arxiv.org/abs/2004.08955) (CVPR 2020)
"""

RSNTREF = """
    Reference:
      - [Deep Residual Learning for Image Recognition](
          https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

setattr(WideResNet50, '__doc__', WideResNet50.__doc__ + WRSNTREF + DOC)
setattr(WideResNet101, '__doc__', WideResNet101.__doc__ + WRSNTREF + DOC)
setattr(ResNeXt50, '__doc__', ResNeXt50.__doc__ + RSNXTREF + DOC)
setattr(ResNeXt101, '__doc__', ResNeXt101.__doc__ + RSNXTREF + DOC)
setattr(ResNeSt50, '__doc__', ResNeXt50.__doc__ + RSNSTREF + DOC)
setattr(ResNeSt101, '__doc__', ResNeXt50.__doc__ + RSNSTREF + DOC)
setattr(ResNeSt200, '__doc__', ResNeXt50.__doc__ + RSNSTREF + DOC)
setattr(ResNeSt269, '__doc__', ResNeXt50.__doc__ + RSNSTREF + DOC)
setattr(ResNet18, '__doc__', ResNet18.__doc__ + RSNTREF + DOC)
setattr(ResNet34, '__doc__', ResNet34.__doc__ + RSNTREF + DOC)
setattr(ResNet50, '__doc__', ResNet50.__doc__ + RSNTREF + DOC)
setattr(ResNet101, '__doc__', ResNet101.__doc__ + RSNTREF + DOC)
setattr(ResNet152, '__doc__', ResNet152.__doc__ + RSNTREF + DOC)
