
# This Python script will perform action recognition on the UCF Sporting Activity Dataset using Computer Vision and Machine Learning
# It uses the HoG feature descriptor to extract features from the frames to create training and testing datasets, and Neural Networks to learn to recognize the action.
# Author: ArvindRS
# Date: 11/30/2016   

import numpy as np, scipy.misc, scipy.signal
import time, os, math, csv, glob


def load_image_with_CV2(filename):

	import cv2

	I = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

	return I

def resize_image(image,new_size):

	import cv2

	I = cv2.resize(image, new_size)

	return I


def get_images(directory_path,resize_size):

	print directory_path

	image_list = []

	for root, sub_folders, files in os.walk(directory_path):
		print directory_path
		print root
		print sub_folders
		print files

		for file in files: 
			if '.jpg' in file:
				file_path = os.path.join(root,file)
				print file_path

				I = load_image_with_CV2(file_path)
				I = resize_image(I,resize_size)

				image_list.append(I)

	return image_list		

def load_frames(image_list,frames_required):

	width = image_list[0].shape[0]
	height = image_list[0].shape[1]
	no_of_frames = len(image_list)
	print width,height,no_of_frames
	space_time_volume = np.zeros((width,height,frames_required))
	print space_time_volume
	print space_time_volume.shape
	step = no_of_frames / frames_required

	index = 0
	for i in range(0,no_of_frames,step):
		print i

		space_time_volume[:,:,index] = image_list[i]
		index += 1
		if index == frames_required:
			break

	print space_time_volume
	print space_time_volume.shape

	return space_time_volume

def get_features(space_time_volume):

	from skimage.feature import hog

	x_range, y_range, z_range = space_time_volume.shape

	hist = []
	print hist

	print x_range,y_range,z_range

	for i in range(z_range):
		frame = space_time_volume[:,:,i]
		print frame
		fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
		print fd
		print fd.shape
		print hog_image
		print hog_image.shape
		hist.append(fd)
	
	print hist

	feature_vector = np.array(hist[0])

	for feature_descriptor in hist:
		feature_vector = np.concatenate((feature_vector,feature_descriptor),axis=0)
	
	print feature_vector
	print feature_vector.shape

	return feature_vector

def write_to_csv(feature_vector,action_class_indicator,output_file):

	row = (list(feature_vector)) + [action_class_indicator]
	print row
	
	with open(output_file,"a") as csvfile:
		csv_pointer = csv.writer(csvfile)
		csv_pointer.writerow(row)

def create_dataset(directory_path,action_class_dict,resize_size,frames_required,output_file):

	import cv2
	from skimage.feature import hog

	fp = open(output_file,"w")

	for key in action_class_dict.keys():
		print key
		action_class_path = directory_path + key + "/"
		print action_class_path

		action_class_indicator = action_class_dict[key]
		print action_class_indicator
		
		for root, sub_folders, files in os.walk(action_class_path):
			for folder in sub_folders:
				if "jpeg" in folder:
					print folder
					print root
					action_class_path_jpeg = os.path.join(root,folder)
					print action_class_path_jpeg
					image_list = get_images(action_class_path_jpeg,resize_size)
					space_time_volume = load_frames(image_list,frames_required)
					feature_vector = get_features(space_time_volume)
					write_to_csv(feature_vector,action_class_indicator,output_file)

	fp.close()

def split_dataset(output_file,training_file,test_file):

	fp = open(output_file,'r')
	temp = open(training_file,'w')
	temp.close()
	temp = open(test_file,'w')
	temp.close()
	training_y_csv = open("training_y.csv",'w')
	test_y_csv = open("test_y.csv",'w')
	training_y_csv.close()
	test_y_csv.close()
	training_fp = open(training_file,'a')
	test_fp = open(test_file,'a')
	training_y_csv = open("training_y.csv",'a')
	test_y_csv = open("test_y.csv",'a')
	csv_reader = csv.reader(fp)
	csv_writer_training = csv.writer(training_fp)
	csv_writer_test = csv.writer(test_fp)
	action_class_indicator = -99
	for row in csv_reader:
		current_row = row
		if action_class_indicator != current_row[-1]:
			action_class_indicator = current_row[-1]
			print action_class_indicator
			csv_writer_test.writerow(current_row)
			test_y_csv.write(action_class_indicator+"\n")
		else:
			csv_writer_training.writerow(current_row)
			training_y_csv.write(action_class_indicator+"\n")
		
	training_y_csv.close()
	test_y_csv.close()
	test_fp.close()
	training_fp.close()
	fp.close()

def run_neural_net(training_file,test_file):

	import pandas as pd
	from sklearn.cross_validation import train_test_split
	from sklearn.neural_network import MLPClassifier
	from sklearn import metrics
	from sklearn.metrics import classification_report,confusion_matrix

	training_csv = pd.read_csv(training_file)
	training_y_csv = pd.read_csv("training_y.csv")
	test_csv = pd.read_csv(test_file)
	test_y_csv = pd.read_csv("test_y.csv")
	X_train, X_test, y_train, y_test = train_test_split(training_csv, training_y_csv, test_size= 0.3, random_state=42) 
	model = MLPClassifier(hidden_layer_sizes=(100,75), activation='relu',random_state = 42)
	model.fit(X_train,y_train)
	y_predicted = model.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_predicted)

	print accuracy

	confusion_mat = confusion_matrix(y_test,y_predicted)

	print confusion_mat
	print confusion_mat.shape

	print "TP\tFP\tFN\tTN\tSensitivity\tSpecificity"
	for i in range(confusion_mat.shape[0]):
		TP = round(float(confusion_mat[i,i]),2)  
		FP = round(float(confusion_mat[:,i].sum()),2) - TP 
		FN = round(float(confusion_mat[i,:].sum()),2) - TP  
		TN = round(float(confusion_mat.sum().sum()),2) - TP - FP - FN
		print str(TP)+"\t"+str(FP)+"\t"+str(FN)+"\t"+str(TN),
		sensitivity = round(TP / (TP + FN),2)
		specificity = round(TN / (TN + FP),2)
		print "\t"+str(sensitivity)+"\t\t"+str(specificity)+"\t\t"

	


def evaluate_dataset(output_file,training_file,test_file):

	#Implements Leave-One-Out cross validation
	split_dataset(output_file,training_file,test_file)

	run_neural_net(training_file,test_file)
	

def main(directory_path,action_class_dict):

	print directory_path

	resize_size = (100,50)
	frames_required = 15
	output_file = "dataset.csv"
	training_file = "training.csv"
	test_file = "test.csv"
	
	#Please download sklearn_18 dev version and comment each of the following function in turn when running the other function 
	create_dataset(directory_path,action_class_dict,resize_size,frames_required,output_file)
	
	evaluate_dataset(output_file,training_file,test_file)


if __name__ == "__main__":
	start_time = time.time()
	current_path = os.getcwd()
	print current_path
	directory_path = "/ucf_sports_actions/ucf action/"
	directory_path = current_path + directory_path
	print directory_path

	action_class_dict = {"Golf-Swing":1,"Kicking":2,"Riding-Horse":3,"Run-Side":4,"SkateBoarding-Front":5,"Swing-Bench":6,"Swing-SideAngle":7,"Walk-Front":8}
	main(directory_path,action_class_dict)
	end_time = time.time()

	print "Runtime : "+str((end_time - start_time) / 60)+" minutes"

