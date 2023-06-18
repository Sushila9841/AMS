# to read video capture bideo and soon
import cv2
# for the interact with current path
import os
import tensorflow as tf
from tensorflow import keras
# for date and time 
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

L_file="FileLabel.txt"
dir_data_="IMAGES_DATASET"

# for reading the label from 
Label_S=[]
with open(L_file,"r") as F:
    Lines = F.readlines()
    for line in Lines:
        Label= line.strip().split(" ")[1]
        Label_S.append(Label)
# to initialize  arrays
DATA=[]
LABELS_ENCODED=[]
# the dataset iteration folders
for xlabel, Folder in enumerate(os.listdir(dir_data_)):
    P_folder = os.path.join(dir_data_,Folder)
    for F_name in os.listdir(P_folder):
        IMAGE_PATH= os.path.join(P_folder,F_name)
        # The images in folder is iterated
        S_img = cv2.imread(IMAGE_PATH)
        S_img = cv2.cvtColor(S_img, cv2.COLOR_BGR2RGB)
        # for resizing image
        S_img = cv2.resize(S_img,(220,220))
        S_img = S_img/255.0
        # storing in array 
        DATA.append(S_img)
        LABELS_ENCODED.append(xlabel)
# for converting data to numpy array  
DATA=np.array(DATA)
LABELS_ENCODED=np.array(LABELS_ENCODED)
# for spliting the data for testing and trainig
X_train,x_test,y_train,y_test=train_test_split(DATA,LABELS_ENCODED,test_size=0.2,random_state=44)
# to define the architecture
MODEL= keras.Sequential([
    keras.applications.MobileNetV2(include_top=False,input_shape=(220,220,3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(len(Label_S),activation="softmax")
])
# the model is compiled
MODEL.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
MODEL.fit(X_train,y_train,epochs=10, batch_size=32, validation_data=(x_test, y_test))
MODEL.save("Kerasdetection.hs")


    

