# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 06:49:25 2020

@author: Admin
"""

import cv2
import time
import numpy as np

#load caffe prototext file.
#Prototxt is a configuration file used to tell caffe how you want the network trained.
#It contains data like the network architecture, learning rate, momentum, weight_decay etc
protoFile = "HandPose/hand/pose_deploy.prototxt"

#Loading the weights of a pre-trained model
weightsFile = "HandPose/hand/pose_iter_102000.caffemodel"

#Pose_Pairs will be used later to communicate which points are to be joined by a line when building the skeleton.
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

#Read deep learning network represented in the prototext configuration file.
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

test_files = ['Test_1.jpg','Test_2.jpg','Test_3.jpg']

time_per_run = []

for test_file in test_files:
    t = time.time()
    print(test_file)
    print(time.asctime(time.localtime(t)))
    
    #Read image
    frame = cv2.imread("Test_Images/"+test_file)
    frameCopy = np.copy(frame)
    
    #Determine height and width of image and get the aspect ratio. 
    #Will be used to resize the image before feeding into the model.
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = 0.1
    # input image dimensions for the network
    inHeight = 350
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    
    #Preprocessing the image by mean subtraction, scaling and optionally channel swapping (False in our case).
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    
    #Set the blob as input for the neural network.
    net.setInput(inpBlob)
    
    print(time.asctime(time.localtime(time.time())))
    #Get predictions through a forward pass of the Neural Net.
    output = net.forward()
    print(time.asctime(time.localtime(time.time())))
    
    # Empty list to store the detected keypoints
    points = []
    
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)
    
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
    
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    
    cv2.imwrite('Test_Results/'+test_file+'-Keypoints.jpg', frameCopy)
    cv2.imwrite('Test_Results/'+test_file+'-Skeleton.jpg', frame)
    
    print(time.asctime(time.localtime(time.time())))
    time_per_run.append(time.time() - t)