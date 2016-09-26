#!/usr/bin/python

#Python implementation of the canny edge detection algorithm
#author: Arvind RS (arvindrs.gb@gmail.com)
#date: 17/09/2016

import numpy as np, scipy.misc, scipy.signal
import time, os, math


def load_image(filename):
	
	#load the image
	I = scipy.misc.imread(filename)

	return I


def smooth_image(I,sigma,k):

	#create the gaussian filter
	gaussian_filter = np.zeros((2*k+1,2*k+1))

	for i in range(1,gaussian_filter.shape[0]+1):
		for j in range(1,gaussian_filter.shape[1]+1):
			gaussian_filter[i-1][j-1] = (1/(2*math.pi*(sigma**2)))*(math.exp((-1*(((i-(k+1))**2)+((j-(k+1))**2)))/(2*(sigma**2))))

	I = scipy.signal.convolve2d(I,gaussian_filter)

	return I

def round_edge_gradient_direction(gradient_direction):
	
	#normalize the angles
	for i in range(gradient_direction.shape[0]):
		for j in range(gradient_direction.shape[1]):
			if (gradient_direction[i][j] >= 0 and gradient_direction[i][j] < 22.5) or (gradient_direction[i][j] >= 157.5 and gradient_direction[i][j] < 202.5) or (gradient_direction[i][j] >= 337.5 and gradient_direction[i][j] <= 360):
				gradient_direction[i][j] = 0
			elif (gradient_direction[i][j] >= 22.5 and gradient_direction[i][j] < 67.5) or (gradient_direction[i][j] >= 202.5 and gradient_direction[i][j] < 247.5):
				gradient_direction[i][j] = 45
			elif (gradient_direction[i][j] >= 67.5 and gradient_direction[i][j] < 112.5) or (gradient_direction[i][j] >= 247.5 and gradient_direction[i][j] < 292.5):
				gradient_direction[i][j] = 90
			else:
				gradient_direction[i][j] = 135

	return gradient_direction

def get_intensity_gradient(I,filename):
	
	#using kernel to extract gradient intensity information
	sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

	Gx = scipy.signal.convolve2d(I,sobel_x,mode='same')
	Gy = scipy.signal.convolve2d(I,sobel_y,mode='same')

	#calculate gradient magnitude
	gradient_magnitude = 	scipy.hypot(Gx,Gy)

	#calculate gradient direction
	gradient_direction = scipy.arctan2(Gy,Gx)

	gradient_direction = round_edge_gradient_direction(gradient_direction)

	return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude,gradient_direction):

	#applying non-maximum suppression algorithm
	output_image = gradient_magnitude

	M = gradient_magnitude.shape[0]
	N = gradient_magnitude.shape[1]

	for i in range(M):
		for j in range(N):
			if gradient_direction[i][j] == 0:
				if (j>0 and gradient_magnitude[i][j] <= gradient_magnitude[i][j-1]) or (j<N-1 and gradient_magnitude[i][j] <= gradient_magnitude[i][j+1]):
					output_image[i][j] = 0
			elif gradient_direction[i][j] == 45:
				if (i>0 and j<N-1 and gradient_magnitude[i][j] <= gradient_magnitude[i-1][j+1]) or (i<M-1 and j>0 and gradient_magnitude[i][j] <= gradient_magnitude[i+1][j-1]):
					output_image[i][j] = 0
			elif gradient_direction[i][j] == 90:
				if (i>0 and gradient_magnitude[i][j] <= gradient_magnitude[i-1][j]) or (i<M-1 and gradient_magnitude[i][j] <= gradient_magnitude[i+1][j]):
					output_image[i][j] = 0
			elif gradient_direction[i][j] == 135:
				if (i>0 and j>0 and gradient_magnitude[i][j] <= gradient_magnitude[i-1][j-1]) or (i<M-1 and j<N-1 and gradient_magnitude[i][j] <= gradient_magnitude[i+1][j+1]):
					output_image[i][j] = 0
		

	return output_image

def hysteresis_thresholding(gradient_magnitude):

	#applying double thresholding
	high_level_threshold = 0.2*np.amax(gradient_magnitude)
	low_level_threshold = 0.1*np.amax(gradient_magnitude)

	M = gradient_magnitude.shape[0]
	N = gradient_magnitude.shape[1]

	output_image = gradient_magnitude.copy()

	for i in range(M):
		for j in range(N):
			if gradient_magnitude[i][j] < low_level_threshold:
				output_image[i][j] = 0
			elif gradient_magnitude[i][j] >= low_level_threshold and gradient_magnitude[i][j] < high_level_threshold:
				if (i>0 and j>0 and gradient_magnitude[i-1][j-1]>0) or (i>0 and gradient_magnitude[i-1][j]>0) or (i>0 and j<N-1 and gradient_magnitude[i-1][j+1]>0) or \
					(j>0 and gradient_magnitude[i][j-1]>0) or (gradient_magnitude[i][j]>0) or (j<N-1 and gradient_magnitude[i][j+1]>0) or \
					(i<M-1 and j>0 and gradient_magnitude[i+1][j-1]>0) or (i<M-1 and gradient_magnitude[i+1][j]>0) or (i<M-1 and j<N-1 and gradient_magnitude[i+1][j+1]>0):
					pass
				else:
					output_image[i][j] = 0

	return output_image

def save_image(filename,image):

	scipy.misc.imsave(filename,image)

def main(sigma,filename):

	current_path = os.getcwd()
	#initialing values
	file_path = current_path + "/" + filename
	k = 2

	#load the image into a matrix I
	I = load_image(file_path)
	filename = filename.replace(".jpg","")
	filename = filename + "_" + str(sigma)

	#smooth the image
	I = smooth_image(I,sigma,k)
	#save_image(filename+"_1_smoothened_image.jpg",I)

	#find intensity gradient of the image
	gradient_magnitude, gradient_direction = get_intensity_gradient(I,filename)
	#save_image(filename+"_2_gradient_mag.jpg",gradient_magnitude)

	#apply non-maximum suppression to the image
	O = non_maximum_suppression(gradient_magnitude,gradient_direction)
	#save_image(filename+"_4_nonmax_suppressed.jpg",O)

	#apply hysteresis thresholding
	O = hysteresis_thresholding(O)
	#save_image(filename+"_5_hysteresis_threshold_applied.jpg",O)

	#save final image to filesystem
	save_image(filename+"_final_output.jpg",O)


start_time = time.time()
filenames = ["lenaTest3.jpg","388016.jpg","277095.jpg"]
sigma = [1.4]
for i in sigma:
	for filename in filenames:
		main(i,filename)
end_time = time.time()

print "Runtime : "+str((end_time - start_time) / 60)+" minutes"
