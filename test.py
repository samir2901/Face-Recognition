import cv2
import os

test_dir = "test_images"
img_list = os.listdir(test_dir)
DATADIR = "Images"
PEOPLES = os.listdir(DATADIR)

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")
img = cv2.imread("test_images/Bale_Test_1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h,x:x+w]
    label, conf = recognizer.predict(roi_gray)
    if conf >= 55:
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = PEOPLES[label] 
        color = (255,255,255)
        stroke = 2
        cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)    
    cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0), 2)
        
img = cv2.resize(img,(500,500))
cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


