# to read video capture bideo and soon
import cv2
# for the interact with current path
import os
from tkinter import messagebox

# =========to choose the user option ==============
User_opt=input("OPTION:\n1. Webcam:\n2. Upload from file:\n")
# Choose to select option

if User_opt=="1": #to capture image from webcam
    # to write the username 
    User_name= input("Kindly share your name with us:")
    # for creating the folder with the help of given Username 
    Folder="IMAGES_DATASET/{}".format(User_name)
    # for giving os path to make folder
    os.makedirs(Folder, exist_ok=True)
    # for caapturing video using webcam
    img_capture=cv2.VideoCapture(0)
    # Pre-trained face detection model of haarcascade is loaded
    Cascade_face=cv2.CascadeClassifier(cv2.data.haarcascade + "haarcascade_frontal_default.xml") 

    _image_no=0
    while True:
        # frame is captured and read 
        F, _frame = img_capture.read()
        # for converting in  frame to grayscale
        F_gray=cv2.cvtColor(_frame,cv2.COLOR_BGR2GRAY)
        # for detecting face in gray scale
        g_face=Cascade_face.detectMultiScale(F_gray, scaleFactor=1.2, minNeighbors=5)
        # for creating rectanlgle box around faces
        for (x,y,w,h) in g_face:
            cv2.rectangle(_frame, (x,y), (x+w, y+h), (1,255,1),2)
            cv2.imshow("Face Cropped", g_face)
            if cv2.waitKey(1) & 0xFF == ord("s")and len(g_face)>0:
                img_path="{}/{:02d}.jpg".format(Folder,_image_no)
                cv2.imwrite(img_path,g_face)            
                messagebox("Saved your Image:", img_path)
                _image_no+=1

        #13 is for enter key which will close 
        if cv2.waitKey(1) & 0xFF == ord("s") or int(_image_no)==100:
             break
    img_capture.release()
    # for destrorying the window
    cv2.destroyAllWindows()

elif User_opt =="2":
    # for the image path
    F_path=input("PLEASE!! Kindly share image file path:")
    # for creating folder with the given name for datasets
    if os.path.isfile(F_path):
        # for adding the name to the folder 
        f_na=input("Kindly give name of the folder:")
        # for creating folder using the user given name 
        U_folder="IMAGES_DATASET/{}".format(f_na) 
        os.makedirs(U_folder, exist_ok=True)
        # for loading the image
        U_image=cv2.imread(F_path)
        # for saving image with incresing number
        Nocount=0
        while Nocount<=100:
            Img_p = "{}/{:02d}.jpg".format(U_folder,Nocount)
            cv2.imwrite(Img_p,U_image)
            ("Successfully!! saved th image", Img_p)
            Nocount+=1
    else:
        print("Sorry!!! couldnt find your file")
else:
    print("Selected option is not valid")
# for updateing label txt
# creating txt file for label 
label_f=open("FileLabel.txt","w")
# getiing list of folder name from image folder 
FolD=os.listdir("IMAGES_DATASET")

#to write the  labels
for l, Fol in enumerate(FolD):
    f_label="{} {}\n".format(l,Fol)
    label_f.write(f_label)
# closeing the folder after creating 
label_f.close()
print("FileLabel.txt successfully generated. CONGRATS")







