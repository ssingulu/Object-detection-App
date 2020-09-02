import streamlit as st
import pandas as pd
from packages import social_distancing_config as config
from packages.Object_detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import io

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
local_css("style.css")
st.title('Object detection')
#disabling warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

labelsPath =os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS =open(labelsPath).read().strip().split("\n")

weightsPath =os.path.sep.join([config.MODEL_PATH,"yolov3.weights"])

configPath= os.path.sep.join([config.MODEL_PATH,"yolov3.cfg"])
print("[INFO] loading YOLO from disk")

#net: The pre-trained and pre-initialized YOLO object detection model
    
net =cv2.dnn.readNetFromDarknet(configPath,weightsPath)

if config.USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


#ln: YOLO CNN output layer names

ln =net.getLayerNames()
ln = [ln[i[0]-1]for i in net.getUnconnectedOutLayers()]
uploaded_image = st.file_uploader("Choose an image...")
objects_list = st.multiselect("Select Objects", LABELS)
social_distancing_check = False
if "person" in objects_list:
    social_distancing_check = st.checkbox("Social distancing check")
print(social_distancing_check)
def predict_objects(uploaded_file):
    frame = imutils.resize(uploaded_file,width =700, height=400)
    print('gowtam')
    print(frame.shape)
    labels=[]
    labels.append(objects_list)
    violations = 0
    for label in objects_list:
        results =detect_people(frame,net,ln,personIdx=LABELS.index(label))
        violate =set()
        
        if len(results) >=2:
            
            # extract all centroids from the results and compute the Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            if social_distancing_check and (label=="person"):
                Cendis = dist.cdist(centroids,centroids,metric="euclidean")
                D = Convertltd([r[1] for r in results])
                
                # Using the widths of bounding boxes and dis btwn their centroids to find violating boxes
                for i in range(len(D)):
                    for j in range(i+1,len(D)):
                        area1 = abs(D[i][3]-D[i][1])*abs(D[i][2]-D[i][0])
                        area2 = abs(D[j][3]-D[j][1])*abs(D[j][2]-D[j][0])
                        print(area1)
                        print(area2)
                        if(abs(area1-area2)/(400*700)<=0.01):
                            if Cendis[i,j] < 75:
                                violate.add(i)
                                violate.add(j)
                        # if (D[i][3]-D[i][1]) >= (D[j][3]-D[j][1]):
                        #     if (D[i][3]-D[i][1])*0.8<=(D[j][3]-D[j][1])<=(D[i][3]-D[i][1]):
                        #         if (D[i][2]-D[i][0]) > (D[j][2]-D[j][0]):
                        #             if (D[j][2]-D[j][0])*0.3<=Cendis[i,j]<=(D[i][2]-D[i][0])*1: #(D[j][2]-D[j][0])<=
                        #                 violate.add(i)
                        #                 violate.add(j)
                        #         else:
                        #             if (D[i][2]-D[i][0])*0.3<=Cendis[i,j]<=(D[j][2]-D[j][0])*1: #(D[i][2]-D[i][0])<=
                        #                 violate.add(i)
                        #                 violate.add(j)
                        # else:
                        #     if (D[j][3]-D[j][1])*0.8<=(D[i][3]-D[i][1])<=(D[j][3]-D[j][1]):
                        #         if (D[i][2]-D[i][0]) >= (D[j][2]-D[j][0]):
                        #             if (D[j][2]-D[j][0])*0.3<=Cendis[i,j]<=(D[i][2]-D[i][0])*1: #(D[j][2]-D[j][0])<=
                        #                 violate.add(i)
                        #                 violate.add(j)
                        #         else:
                        #             if (D[i][2]-D[i][0])*0.3<=Cendis[i,j]<=(D[j][2]-D[j][0])*1: #(D[i][2]-D[i][0])<=
                        #                 violate.add(i)
                        #                 violate.add(j)
            
    # loop over the upper triangular of the distance matrix
        for(i,(prob,bbox,centroid)) in enumerate(results):
            (startX,startY,endX,endY)=bbox
            (cX,cY)=centroid 
            color=(0,255,0)

            if social_distancing_check and (i in violate) :
                color = (0,0,255)
                violations += 1
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
            #cv2.circle(frame,(cX,cY),5,color,1)
        st.header('Number of detected '+str(label)+': '+str(len(results)))
        if label=="person" and social_distancing_check and violations>0:
            t = "<div> <span class='highlight blue'>Number of people violating social distancing: <span class='bold'>{}</span> </span></div>"
            st.markdown(t.format(violations), unsafe_allow_html=True)
    st.subheader("Image after object detection")

    st.image(frame, channels="BGR")
def Convertltd(lst): 
	res_dct = {i: lst[i] for i in range(0, len(lst))} 
	return res_dct

if st.button('Predict the objects'):
    print(objects_list)
    if uploaded_image is not None:
        #uploaded_image = io.TextIOWrapper(uploaded_image)
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.subheader("original image")
        initial_image = imutils.resize(opencv_image, width =700, height=400)
        st.image(initial_image, channels="BGR")
        predict_objects(opencv_image)