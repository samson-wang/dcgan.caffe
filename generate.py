import caffe
import numpy as np
import sys
import cv2
import scipy.io
import scipy.misc

nz = 100
img_size = 64
batch_size = 64

caffe.set_mode_gpu()
gen_net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)

# Fix the seed to debug
np.random.seed(0) 
gen_net.blobs['feat'].data[...] = np.random.normal(0, 1, (batch_size, nz)).astype(np.float32)

gen_net.forward_simple()

generated_img = gen_net.blobs['generated'].data

print generated_img.shape

print generated_img[0].transpose(1,2,0)
max_val, min_val = np.max(generated_img[0]), np.min(generated_img[0])

#matfile = scipy.io.loadmat('/data/Repo/release_deepsim_v0.5_train/trained_models/caffenet/ilsvrc_2012_mean.mat')
#image_mean = matfile['image_mean']
#print image_mean.shape, image_mean, image_mean[14:241, 14:241, :] + generated_img[0].transpose(1,2,0)

# Concat all images into a big 8*8 image
flatten_img = ((generated_img.transpose((0,2,3,1)))[:] - min_val) / (max_val-min_val)
print flatten_img.shape
#print flatten_img.reshape(2, 2, 64, 64, 3).shape
#scipy.misc.imsave('test1.png', flatten_img.reshape(8,8,img_size,img_size,3).swapaxes(1,2).reshape(8*img_size,8*img_size, 3))
cv2.imshow('test1', flatten_img.reshape(8,8,img_size,img_size,3).swapaxes(1,2).reshape(8*img_size,8*img_size, 3))
cv2.waitKey()

#cv2.imshow('test', ((generated_img.transpose((0,2,3,1)))[2] - min_val) / (max_val-min_val))
#cv2.waitKey()
