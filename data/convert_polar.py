# Code source: https://github.com/shiyujiao/cross_view_localization_SAFA/blob/master/script/data_preparation.py
import numpy as np
import imageio
import os
from PIL import Image
from matplotlib import pyplot


def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
    fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11

    return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2


############################ Apply Polar Transform to Aerial Images in CVUSA Dataset ############################
def polar_transform_CVUSA(input_dir, output_dir):
  S = 750
  height = 112
  width = 616

  i = np.arange(0, height)
  j = np.arange(0, width)
  jj, ii = np.meshgrid(j, i)

  y = S / 2. - S / 2. / height * (height - 1 - ii) * np.sin(2 * np.pi * jj / width)
  x = S / 2. + S / 2. / height * (height - 1 - ii) * np.cos(2 * np.pi * jj / width)

  # input_dir = './placeholder_bingmap/'
  # output_dir = './placeholder_polarmap/'

  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  images = os.listdir(input_dir)
  

  for i, img in enumerate(images):
    if i % 1000:
        print('image=', i)
    signal = imageio.imread(input_dir + img)
    image = sample_bilinear(signal, x, y)
    # img_uint = image.astype(np.uint8)
    imageio.imsave(output_dir + img,  image)
    

############################ Apply Polar Transform to sat Images in VIGOR Dataset ############################
def polar_transform_VIGOR(input_dir, output_dir):

  S = 640
  height = 320
  width = 640

  i = np.arange(0, height)
  j = np.arange(0, width)
  jj, ii = np.meshgrid(j, i)

  y = S / 2. - S / 2. / height * (height - 1 - ii) * np.sin(2 * np.pi * jj / width)
  x = S / 2. + S / 2. / height * (height - 1 - ii) * np.cos(2 * np.pi * jj / width)

  # input_dir = 'VIGOR/Chicago/satellite/'
  # output_dir = 'VIGOR/Chicago/polar/'

  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  images = os.listdir(input_dir)
  

  for i, img in enumerate(images):
    if i // 1000:
        print('**********image=', i)
    signal = imageio.imread(input_dir + img)
    image = sample_bilinear(signal, x, y)
    # img_uint = image.astype(np.uint8)
    imageio.imsave(output_dir + img,  image)


if __name__ == '__main__':
    polar_transform_VIGOR('VIGOR/Chicago/satellite/', 'VIGOR/Chicago/polar/')