# extract deep features for all the images in this directory
import caffe
import numpy as np
import sys
import os
import scipy
caffe.set_mode_cpu()

net = caffe.Net('/media/martin/MartinK3TB/Documents/caffe_new/models/bvlc_reference_caffenet/deploy.prototxt', '/media/martin/MartinK3TB/Documents/caffe_new/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('/media/martin/MartinK3TB/Documents/caffe_new/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(1,3,227,227)

#you can print available layers and their sizes with: [(k, v.data.shape) for k, v in net.blobs.items()]

print('starting to itterate through images')

for line in open("extracted_images.txt"):
	print(line.strip())
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(line.strip()))
	out = net.forward()
	if 'feature_matrix' in locals():
		new_matrix=np.matrix(net.blobs['pool5'].data.reshape((1, np.prod(net.blobs['pool5'].data.shape))))#(net.blobs['fc6'].data)
		feature_matrix = np.vstack([feature_matrix, new_matrix])
	else:
		feature_matrix=np.matrix(net.blobs['pool5'].data.reshape((1, np.prod(net.blobs['pool5'].data.shape))))#net.blobs['fc6'].data)

np.save('features_pool5.npy', feature_matrix)
#Returns a condensed distance matrix Y. For each i and j (where i<j<n), the metric dist(u=X[i], v=X[j]) is computed and stored in entry ij.
image_distances=scipy.spatial.distance.pdist(feature_matrix, 'cityblock')#also 'euclidean'
np.save('image_distances_pool5.npy', image_distances)

np.set_printoptions(precision=3)
print(image_distances)
print(scipy.spatial.distance.squareform(image_distances))