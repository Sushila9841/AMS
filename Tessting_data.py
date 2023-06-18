# to read video capture bideo and soon
import cv2
# for the interact with current path
import os
from sklearn import model_selection
import tensorflow as tf
from tensorflow import keras
# for date and time 
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

# loaddataset
Logged_attendance="Attend.txt"
# function for get class name 
def get_class_name(class_no):
    if class_no==0:
        return"Bipana"
    if class_no==1:
        return"Nishita"
    if class_no==2:
        return"Rovika"
    if class_no==3:
        return"Sadikshya"
    if class_no==4:
        return"Shanti"
    if class_no==5:
        return"Sushila"
# loading pretrained model of keras    # 
MODEL=keras.models.load_model("Kerasdetection.hs")

# =====for test Data======
# for loading the pretained data for face detection from Haarcascade
Face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades +"haarcascade_frontal_default.xml")
Capture=cv2.VideoCapture(0)
Capture.set(3,640)
Capture.set(4,480)
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    S,Oringin_Img=Capture.read()
    # for converting in  frame to grayscale
    F_gray=cv2.cvtColor(Oringin_Img, cv2.COLOR_BGR2GRAY)
    # for detecting face in gray scale
    g_face=Face_cascade.detectMultiScale(F_gray,1.3,5)

    for (X,Y,W,H) in g_face:
            cv2.rectangle(Oringin_Img, (X,Y), (X+W, Y+H), (1,255,1),2)
            Imgcrop=Oringin_Img[Y:Y+H,X:X+W]
            R_Img=cv2.resize(Imgcrop,(220,220))
            R_Img= R_Img/255.0
            R_Img=np.expand_dims(R_Img,axis=0)
            Predict= MODEL.predict(R_Img)
            Index_C=np.argmax(Predict)
            Name_C=get_class_name(Index_C)

            # putting text
            cv2.putText(Oringin_Img,Name_C, (X,Y+20),font,0.75,(0,255,1),1,cv2.LINE_AA)
            with open(Logged_attendance,"b") as F:
                # for time 
                TimeStamp=datetime.datetime.now().strftime("%d,%m,%Y %H:%M:%S")
                Entry_attend=f"{TimeStamp}-{Name_C}\n"
                F.write(Entry_attend)
            # to display the attendance of the student
            cv2.putText(Oringin_Img,"Registration successfully done",(10,20),font,0.75,(1,255,1),2,cv2.LINE_AA)
        # to show the data 
    cv2.imshow("R",Oringin_Img)
    cv2.waitKey(1)
    if len(g_face)>0:
        # time to take face
        cv2.waitKey(1)==13
        break
Capture.release()
# destroythe all windows
cv2.destroyAllWindows()






    
    
        