from tensorflow import image as utils
from tensorflow.keras.applications import imagenet_utils


def preprocess(x, data_format=None):
  """Preprocesses images for ResNet models.

  Order of preprocessing:
    - Resize to 256 by 256
    - Central crop to 224 by 224
    - Normalize values by scaling into the range [0, 1]
    - Normalize values by mean subtraction
    - Normalize values by dividing by the standard deviation
  """
  x = utils.resize(x, (256, 256))
  x = utils.central_crop(x, (224/256))
  return imagenet_utils.preprocess_input(
      x, data_format=data_format, mode='torch')
