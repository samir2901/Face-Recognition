import os
import numpy as np
import cv2
import pickle 
import time


DATADIR = "Images"
PEOPLES = os.listdir(DATADIR)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
IMAGE_SIZE = (550,550)
x_train = []
y_train = []

t = time.time()

for people in PEOPLES:
    path = os.path.join(DATADIR,people)    
    for image in os.listdir(path):
        if image.endswith("jpg") or image.endswith("png"):
            label = PEOPLES.index(people)
            image_path = os.path.join(path,image)
            #print(image_path,":",label)
            try:
                img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,IMAGE_SIZE)
                img_array = np.array(img,"uint8") 
                faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)
                for (x,y,w,h) in faces:
                    roi = img_array[y:y+h,x:x+h]
                    x_train.append(roi)
                    y_train.append(label)
            except Exception as e:
                pass


print("Training started...")
recognizer.train(x_train,np.array(y_train))
recognizer.save("train.yml")
print("Training DONE")
print("Time taken: {}".format(time.time()-t))




          




