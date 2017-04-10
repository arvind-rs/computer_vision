
Action Recognition using Computer Vision and Machine Learning:

Dataset is available at:
http://crcv.ucf.edu/data/UCF_Sports_Action.php

Action Recognition is the process of recognizing human activity using a series of observations of the human subject and the surrounding environments. The objective of this project was to develop a software system to recognize human activity from the UCF Sporting Activity Dataset using a feature descriptor and a machine learning classifier.

Number of stages:
1. Feature extraction
2. Machine Learning Classifier
3. Evaluation method

Feature Extraction:
The image dataset was first preprocessed and normalized by resizing the image to 100 height * 50 width. To reduce the computational costs, only 15 frames were selected that were evenly picked from each video sequences of the action classes. The HOG Descriptor is used for feature extraction. The Histogram of Oriented Gradients (HOG) works by counting the occurrences of gradient orientation in localized portions of the Image. The number of orientations were set to 8, pixels per cell to 16*16 and cells per block to 1*1. This results in extracting of 2305 features per spatio-temporal volume, i.e., per video sequence. Also the Diving and Lifting action classes were dropped due to lack of annotations in the video sequences.

Machine Learning Classifier:
The machine learning classifier used for action recognition here is Neural Networks. The dataset was split into training set and test set in the ratio 7:3. The MLPClassifier from the sklearn library was used with 2 hidden layers of size (100,75) and the activation function used was the Rectified Linear Unit (ReLU).

Evaluation method:
The performance of this system was evaluated using the sklearn.metrics library. The performance of the system is given below:
Runtime: 0.39 minutes
Accuracy: 81%

Sensitivity and Specificity:
TP 		FP 		FN 		TN 		Sensitivity		Specificity
6.0		0.0		0.0		31.0	1.0				1.0
4.0		1.0		2.0		30.0 	0.67			0.97
2.0		3.0		0.0		32.0 	1.0				0.91
2.0		0.0		2.0		33.0 	0.5				1.0
3.0		1.0		1.0		32.0 	0.75			0.97
5.0		2.0		1.0		29.0 	0.83			0.94
1.0		0.0		1.0		35.0 	0.5				1.0
7.0		0.0		0.0		30.0 	1.0				1.0

