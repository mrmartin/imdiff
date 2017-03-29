#find and show the most similar images, for visual analysis
import numpy as np
from PIL import Image
import scipy.spatial

lines = [line.strip() for line in open('extracted_images.txt')]

#change this to whichever feature you extracted
image_distances = np.load('image_distances_pool5.npy')

image_distances=scipy.spatial.distance.squareform(image_distances);
#find minimum while ignoring the zeros on the diagonal
while True:
	index=np.unravel_index(np.argmin(image_distances+np.max(image_distances)*np.eye(len(lines))), image_distances.shape)

	print(lines[index[0]])
	im = Image.open(lines[index[0]])
	im.show()
	print(lines[index[1]])
	im = Image.open(lines[index[1]])
	im.show()

	image_distances[index]=np.max(image_distances)

	index=np.unravel_index(np.argmin(image_distances+np.max(image_distances)*np.eye(len(lines))), image_distances.shape)

	image_distances[index]=np.max(image_distances)

	raw_input()
