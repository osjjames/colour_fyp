# Code modified from https://github.com/richzhang/colorization/blob/master/colorization/colorize.py

import numpy as np
import os
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
import caffe
import argparse
import time

parent_path = '/src/zhang/'

class Options():
  def __init__(self, img_out, gpu, prototxt, caffemodel):
    self.img_out = img_out
    self.gpu = gpu
    self.prototxt = prototxt
    self.caffemodel = caffemodel

arg_defaults = Options('', 0, parent_path + 'models/colorization_deploy_v2.prototxt', '/resources/colorization_release_v2.caffemodel')

# caffe.set_mode_gpu()
# caffe.set_device(options.gpu)

# Select desired model
net = caffe.Net(arg_defaults.prototxt, arg_defaults.caffemodel, caffe.TEST)

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

pts_in_hull = np.load(parent_path + 'resources/pts_in_hull.npy') # load cluster centers
net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel

def parse_args():
    parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
    parser.add_argument('-img_in',dest='img_in',help='grayscale image to read in', type=str)
    parser.add_argument('-img_out',dest='img_out',help='colorized image to save off', type=str)
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=arg_defaults.gpu)
    parser.add_argument('--prototxt',dest='prototxt',help='prototxt filepath', type=str, default=arg_defaults.prototxt)
    parser.add_argument('--caffemodel',dest='caffemodel',help='caffemodel filepath', type=str, default=arg_defaults.caffemodel)

    args = parser.parse_args()
    return args

def colorize_from_file(img_path, lab_only = False, options = arg_defaults):
  # load the original image
  img_in = caffe.io.load_image(img_path)
  return colorize(img_in, lab_only, options)

def colorize_from_grayscale(img_gray, lab_only = False, options = arg_defaults):
  img_gray = np.divide(img_gray, 256)
  img = np.stack((img_gray,img_gray,img_gray), axis=2)
  return colorize(img, lab_only, options)


def colorize(img_in, lab_only = False, options = arg_defaults):

  # Working within scikit Lab space: L = (0, 100), a,b = (-128, 127)
  # For reference, opencv Lab space: L,a,b = (0, 255)
  if lab_only:
    img_lab = img_in.copy()
    img_lab = (img_lab + [0, -128, -128]) * [100/255., 1, 1]
  else:
    img_rgb = img_in.copy()
    img_lab = color.rgb2lab(img_rgb) # convert image to lab color space

  img_l = img_lab[:,:,0] # pull out L channel
  (H_orig,W_orig) = img_lab.shape[:2] # original image size

  # resize image to network input size
  if lab_only:
    img_lab_rs = caffe.io.resize_image(img_lab,(H_in,W_in)) # resize image to network input size
  else:
    img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
    img_lab_rs = color.rgb2lab(img_rs)
  img_l_rs = img_lab_rs[:,:,0]

  net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
  net.forward() # run network


  ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
  ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
  img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
  if lab_only:
    img_lab_out = (img_lab_out + [0, 128, 128]) * [255/100., 1, 1]
    return img_lab_out.astype('uint8')
    
  else:
    img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb
    return img_rgb_out

if __name__ == '__main__':
  args = parse_args()

  options = Options(args.img_out, args.gpu, args.prototxt, args.caffemodel)

  img_rgb_out = colorize(args.img_in, options)

  plt.imsave(args.img_out, img_rgb_out)

  
