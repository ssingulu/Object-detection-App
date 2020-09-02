# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:47:52 2020

@author: paras
"""


from .social_distancing_config import MIN_CONFID 
from .social_distancing_config import NMS_THRESHLD

import numpy as np
import cv2

#makeing frames from social_detection
def detect_people(frame, net, ln, personIdx=0):
    #grabbing the dimensions of the frame and initializing the list results
    #net: The pre-trained and pre-initialized YOLO object detection model
    #ln: YOLO CNN output layer names

    (H, W) = frame.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    #net.setInput(blob) - passing the images to the algorithm or you can say into the network
    #net.forward(ln) – It will return us the bounded boxes coordinate, centroids and confidence values from the image. 

    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            #confidence, x, y, h, w, pr(1), pr(2), pr(3)
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONFID:
                
                #scale the bounding box coordinates back relative to the size of the image
                
                centerX=int(detection[0]*W)
                centerY=int(detection[1]*H)
                width=int(detection[2]*W)
                height=int(detection[3]*H)
                
                #box = detection[0:4] * np.array([W, H, W, H])
                #(centerX, centerY, width, height) = box.astype("int")

                #use the center (x, y)-coordinates to derive the top and left corner of the bounding box

                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    
    #Non Maxima Supression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFID, NMS_THRESHLD)
    
    if len(idxs) > 0:
        #loop over the indexes we are keeping
        for i in idxs.flatten():
            #extracting bounding box coordinates
            
            x, y, w, h = boxes[i]
            
            #(x, y) = (boxes[i][0], boxes[i][1])
            #(w, h) = (boxes[i][2], boxes[i][3])
            
            #updating our Results list = consisting persons
            #predicting probability, Bounding Box coordinates, centriod
            
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
        
#return the list of results
    return results
            
    
# WORKING....
#frames get preprocessed and are given back to the model and gets o/p fromit
#compared based only on people/persons and returned
#NMS - NON MAXIMA SUPRESSION
#gives back CENTRIOD, BOUNDING BOX CORDINATES, CONFIDENCE

