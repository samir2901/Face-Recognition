import cv2 
import os

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")
DATADIR = "Images"
PEOPLES = os.listdir(DATADIR)
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()    
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        label, conf = recognizer.predict(roi_gray)        

        if conf>=55:            
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = PEOPLES[label] 
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2)
        

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
