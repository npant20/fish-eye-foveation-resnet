import tensorflow as tf
import numpy as np
from skimage.util import crop, pad
from skimage.transform import resize, warp
from scipy.optimize import curve_fit, brenth
from functools import partial

npa = np.array
slim = tf.contrib.slim

def tf_quad_func(x, func_pars):
  return func_pars[0] * x ** 2 + func_pars[1] * x


def tf_exp_func(x, func_pars):
  return tf.exp(func_pars[0] * x) + func_pars[1]


def tf_image_translate(images, t, interpolation='NEAREST'):
  transforms = [1, 0, -t[0], 0, 1, -t[1], 0, 0]
  return tf.contrib.image.transform(tf.expand_dims(images, 0), transforms, interpolation)[0]


def tf_inv_quad_func(x, func_pars):
  a = func_pars[0]
  b = func_pars[1]
  return (-b + tf.sqrt(b ** 2 + 4*a*x))/(2*a)


def sampling_mismatch(rf, in_size=None, out_size=None, max_ratio=10.):
  """
  This function returns the mismatch between the radius of last sampled point and the image size.
  """

  if out_size is None:
    out_size = in_size
  r_max = in_size // 2
  # Exponential relationship
  a = np.log(max_ratio) / 32.0
  r, d = [0.], []
  for i in range(1, out_size // 2):
    d.append(1. / np.sqrt(np.pi * rf) * np.exp(a * r[-1] / 2.))
    r.append(r[-1] + d[-1])
  r = np.array(r)

  return 32.0 - r[-1]


def get_rf_value(input_size, output_size, rf_range=(0.01, 5.)):
  """
  The RF parameter should be tuned in a way that the last sample would be taken from the outmost pixel of the image.
  This function returns the mismatch between the radius of last sampled point and the image size. We use this function
  together with classic root finding methods to find the optimal RF value given the input and output sizes.
  """

  func = partial(sampling_mismatch, in_size=input_size, out_size=output_size)
  return brenth(func, rf_range[0], rf_range[1])


def get_foveal_density(output_image_size, input_image_size):
    return get_rf_value(input_image_size, output_image_size)
  
 
def delta_lookup(in_size, out_size=None, max_ratio=10.):
  """
  Divides the range of radius values based on the image size and finds the distances between samples
  with respect to each radius value. Different function types can be used to form the mapping. All function
  map to delta values of min_delta in the center and max_delta at the outmost periphery.
  :param in_size: Size of the input image
  :param out_size: Size of the output (retina) image
  :param max_ratio: ratio between density at the fovea and periphery
  :return: Grid of points on the retinal image (r_prime) and original image (r)
  """
  rf = get_foveal_density(out_size, in_size)
  if out_size is None:
    out_size = in_size
  r_max = 32.0

  # Exponential relationship
  a = np.log(max_ratio) / r_max
  r, d = [0.], []
  for i in range(out_size // 2):
    d.append(1. / np.sqrt(np.pi * rf) * np.exp(a * r[-1] / 2.))
    r.append(r[-1] + d[-1])
  r = np.array(r)
  r_prime = np.arange(out_size // 2)

  return r_prime, r[:-1]


def fit_func(func, r, r_raw):
  """
  Fits a function to map the radius values in the
  :param func: function template
  :param r: Inputs to the function (grid points on the retinal image)
  :param r_raw: Outputs for the function (grid points on the original image)
  :return: Estimated parameters, estimaged covariance of parameters
  """
  popt, pcov = curve_fit(func, r, r_raw, p0=[0, 0.4], bounds=(0, np.inf))
  return popt, pcov

def find_retina_mapping(input_size, output_size, fit_mode='quad'):
  """
  Fits a function to the distance data so it will map the outmost pixel to the border of the image
  :param fit_mode:
  :return:
  """
  r, r_raw = delta_lookup(in_size=input_size, out_size=output_size)
  if fit_mode == 'quad':
    func = lambda x, a, b: a * x ** 2 + b * x
    tf_func = tf_quad_func
  elif fit_mode == 'exp':
    func = lambda x, a, b: np.exp(a * x) + b
    tf_func = tf_exp_func
  else:
    raise ValueError('Fit mode not defined. Choices are ''linear'', ''exp''.')
  popt, pcov = fit_func(func, r, r_raw)

  return popt, tf_func


def warp_func(xy, orig_img_size, func, func_pars, shift, dxc = 0, dyc = 0):
  # Centeralize the indices [-n, n]
  xy = tf.cast(xy, tf.float32)
  center = tf.reduce_mean(xy, axis=0)
  center_shift = tf.cast(tf.constant([[dxc, dyc]]), tf.float32)
  xy_cent = xy - center - center_shift
  print(xy_cent)

  # Polar coordinates
  r = tf.sqrt(xy_cent[:, 0] ** 2 + xy_cent[:, 1] ** 2)
  theta = tf.atan2(xy_cent[:, 1], xy_cent[:, 0])
  oldr = r
  r = func(r, func_pars)
  ratio = r/oldr

  xs = r * tf.cos(theta)
  xs = xs + tf.math.multiply(ratio, dxc)
  xs += orig_img_size[0] / 2. - shift[0]
  # Added + 2.0 is for the additional zero padding
  xs = tf.minimum(orig_img_size[0] + 2.0, xs)
  xs = tf.maximum(0., xs)
  xs = tf.round(xs)

  ys = r * tf.sin(theta)
  ys = ys + tf.math.multiply(ratio, dyc)
  ys += orig_img_size[1] / 2 - shift[1]
  ys = tf.minimum(orig_img_size[1] + 2.0, ys)
  ys = tf.maximum(0., ys)
  ys = tf.round(ys)

  xy_out = tf.stack([xs, ys], 1)

  xy_out = tf.cast(xy_out, tf.int32)
  return xy_out


def warp_image(img, output_size, input_size=None, shift=None, dxc = 0, dyc = 0):
  """

  :param img: (tensor) input image
  :param retina_func:
  :param retina_pars:
  :param shift:
  :return:
  """
  original_shape = img.shape

  if input_size is None:
    input_size = np.min([original_shape[0], original_shape[1]])

  retina_pars, retina_func = find_retina_mapping(input_size, output_size)

  if shift is None:
    shift = [tf.constant([0], tf.float32), tf.constant([0], tf.float32)]
  else:
    assert len(shift) == 2
    shift = [tf.constant([shift[0]], tf.float32), tf.constant([shift[1]], tf.float32)]
  paddings = tf.constant([[2, 2], [2, 2], [0, 0]])
  img = tf.pad(img, paddings, "CONSTANT")
  row_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=-1), [1, output_size])
  row_ind = tf.reshape(row_ind, [-1, 1])
  col_ind = tf.tile(tf.expand_dims(tf.range(output_size), axis=0), [1, output_size])
  col_ind = tf.reshape(col_ind, [-1, 1])
  indices = tf.concat([row_ind, col_ind], 1)
  xy_out = warp_func(indices, tf.cast(original_shape, tf.float32), retina_func, retina_pars, shift, dxc, dyc)

  out = tf.reshape(tf.gather_nd(img, xy_out), [output_size, output_size, 3]) 
  return out

