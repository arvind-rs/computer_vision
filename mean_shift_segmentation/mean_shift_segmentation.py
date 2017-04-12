#!/usr/bin/python

#Implementing image segmentation using mean shift clustering approach.
#Author: Arvind RS
#Date: 10/21/2016

import numpy as np, scipy.misc, scipy.signal
import time, os, math, sys
import matplotlib.pyplot as plt

#Algorithm:
#1. Load the input image into a variable X.
#2. For each pixel x in X:
#2.a Find all the neighbours of x as neighbours[].
#2.b calculate the mean shift for x and its neighbours.
#2.c assign x with the calculated mean shift value
#3. write X to the filesystem.

def load_image(filename):
	#Load the image
	I = scipy.misc.imread(filename)

	return I

def get_euclidean_distance_vectorized(X,x):
	#function to calculate and return the distance between points in a vector

	return np.sqrt((X - x)**2)

def get_neighbours(x,X,distance):
	#function to extract the points whose intensities are within the neighbourhood of the current point's intensity 
	neighbours = []
	distance_vector = get_euclidean_distance_vectorized(X,x)
	neighbours = np.extract(distance_vector<=distance,X)

	return neighbours

def apply_gaussian_kernel_vectorized(distance_vector,sigma):
	#function that appies the gaussian kernel and returns the value
	temp = distance_vector / sigma
	
	return (np.exp(-5*(temp)**2))

def mean_shift(image,sigma,distance,N):
	#apply mean shift algorithm
	shape = image.shape
	print shape,sigma,distance,N
	X = image.reshape([-1])
	print X.shape
	X_copy = np.copy(X)
	for iteration in range(N):
		for i,x in enumerate(X_copy):

			#find the neighbours around x
			neighbours = get_neighbours(x,X,distance)
			#calculate the mean shift for the neighbours
			numerator = 0
			denominator = 0
			distance_vector = get_euclidean_distance_vectorized(neighbours,x)
			weights = apply_gaussian_kernel_vectorized(distance_vector,sigma)
			numerator = np.sum(weights * neighbours)
			denominator = np.sum(weights)
			mean_shift_value = numerator / denominator
			#update x value with mean shift value
			X_copy[i] = mean_shift_value

	O = X_copy.reshape(shape)

	return O

def save_image(filename,image):
	#function to save the output to the filesystem
	scipy.misc.imsave(filename,image)

def main(filename):
	#main function

	current_path = os.getcwd()
	
	#initializing values
	file_path = current_path + "/" + filename

	#load the image into a matrix I
	I = load_image(file_path)

	print I

	#decided on this hs value by using the estimate_bandwidth() from sklearn.cluster
	hs = 10.86
	hr = 5
	no_of_iterations = 1

	print "running mean shift algorithm..."
	O = mean_shift(I,hs,hr,no_of_iterations)
	print O

	print "saving output to file..."
	filename = filename.replace(".jpg","")
	save_image(filename+"_output.jpg",O)

if __name__ == "__main__":
	start_time = time.time()
	if len(sys.argv) < 2:
		print "Insufficient arguments. Exiting!"
		exit(0)
	arg_input = sys.argv[1]
	#filename = "1.jpg"
	filename = arg_input
	main(filename)
	end_time = time.time()
	print "Runtime : "+str((end_time - start_time) / 60)+" minutes"
