# https://youtu.be/6pUSZgPJ3Yg
"""
Satellite image to maps image translation ​using Pix2Pix GAN
 
Data from: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
Also find other datasets here: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
"""

import matplotlib.pyplot as plt
from os import listdir
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
#from matplotlib import plt
import numpy as np
import os

# load all images in a directory into memory
def load_images(path, size=(256,512)):
	noisy_src_list, clean_tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		file_path = os.path.join(path, filename)
		pixels = load_img(file_path, target_size=size)
		#pixels = load_img(path + filename, target_size=size)

		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		noisy_img, clean_img = pixels[:, 256:], pixels[:, :256]
		noisy_src_list.append(noisy_img)
		clean_tar_list.append(clean_img)
	return [asarray(noisy_src_list), asarray(clean_tar_list)]

# dataset path
path = '/data/home/vvaibhav/AI/Internship/Dataset/PreparedData/MergedSpectograms/SNR_5dB_-15dB/train'
# load dataset
[noisy_src_images, clean_tar_images] = load_images(path)
print('Loaded: ', noisy_src_images.shape, clean_tar_images.shape)


# n_samples = 3
# for i in range(n_samples):
# 	plt.subplot(2, n_samples, 1 + i)
# 	plt.axis('off')
# 	plt.imshow(noisy_src_images[i].astype('uint8'))
# # plot target image
# for i in range(n_samples):
# 	plt.subplot(2, n_samples, 1 + n_samples + i)
# 	plt.axis('off')
# 	plt.imshow(clean_tar_images[i].astype('uint8'))
# plt.show()

#######################################

from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
image_shape = noisy_src_images.shape[1:]
print("image_shape", image_shape)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#Define data
# load and prepare training images
data = [noisy_src_images, clean_tar_images]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

from datetime import datetime 
#from keras.models import save_model

start1 = datetime.now() 

train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=1) 
# #Reports parameters for each batch (total 1096) for each epoch.
# #For 10 epochs we should see 10960
# gan_model.save('saved_model_after_training.h5')

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

# #Reports parameters for each batch (total 1096) for each epoch.
# #For 10 epochs we should see 10960

# #################################################

# #Test trained model on a few images...

# from keras.models import load_model
# from numpy.random import randint
# import cv2

# model = load_model('model_000400.h5')

# # plot source, generated and target images
# # def plot_images(src_img, gen_img, tar_img):
# # 	images = vstack((src_img, gen_img, tar_img))
# # 	# scale from [-1,1] to [0,1]
# # 	images = (images + 1) / 2.0
# # 	titles = ['Source', 'Generated', 'Expected']
# # 	# plot images row by row
# # 	for i in range(len(images)):
# # 		# define subplot
# # 		plt.subplot(1, 3, 1 + i)
# # 		# turn off axis
# # 		plt.axis('off')
# # 		# plot raw pixel data
# # 		plt.imshow(images[i])
# # 		# show title
# # 		plt.title(titles[i])
# # 	plt.show()

# def plot_images(src_img, gen_img, tar_img, save_path=None):
#     images = vstack((src_img, gen_img, tar_img))
#     # scale from [-1,1] to [0,1]
#     images = (images + 1) / 2.0
#     titles = ['Source', 'Generated', 'Expected']
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # plot images row by row
#     for i in range(len(images)):
#         # define subplot
#         ax = axes[i]
#         # turn off axis
#         ax.axis('off')
#         # plot raw pixel data
#         ax.imshow(images[i])
#         # show title
#         ax.set_title(titles[i])
#         # explicitly remove overlapping axes
#         ax.remove()

#     if save_path:
#         plt.savefig(save_path)

#     plt.show()
    
# [X1, X2] = dataset
# # select random example
# ix = randint(0, len(X1), 1)
# src_image, tar_image = X1[ix], X2[ix]
# # generate image from source
# gen_image = model.predict(src_image)
# # plot all three images
# save_path = "/data/home/vvaibhav/AI/Internship/Notebook/Notebook/GANs/PIx2Pix"

# plot_images(src_image, gen_image, tar_image , save_path)


