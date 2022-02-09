import pickle
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name":1}
with open("labels.pickle","rb") as f:
    org_labels = pickle.load(f)
    labels = {v:k for k,v in org_labels.items()}  #inverting from id to name#


cap = cv2.VideoCapture(0)#capturing the video frm src#

while(True):
    ret , frame = cap.read()#reading the feed#
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]

        id_ , conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (233,233,233)
            stroke = 1
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_name = "gray.png"
        cv2.imwrite(img_name,roi_gray)



        color_rectangle = (255,0,0)
        stroke = 2
        end_cord_y = y+h
        end_cord_x = x+w
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y),color_rectangle,stroke)


    cv2.imshow('Frame',frame) #showing the frame#
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
