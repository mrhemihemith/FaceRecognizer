import cv2
#import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
colour_image = cv2.imread('1.jpg')
gray_image = cv2.cvtColor(colour_image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('frame',gray_image)
#cv2.waitKey()
#print(f"gray : {gray_image}")
#print(f"colour: {colour_image}")

faces = face_cascade.detectMultiScale(colour_image)
print(faces)
for (x,y,w,h) in faces:
    roi_colour = colour_image[y:y+h,x:x+w]
    roi_gray = gray_image[y:y+h,x:x+w]
    print(roi_gray)



    color = (255,0,0)
    stroke = 2
    end_cordinate_x = x+w
    end_cordinate_y = y+h
    output = cv2.rectangle(colour_image,(x,y),(end_cordinate_x,end_cordinate_y),color,stroke)
    print(output)
    cv2.imshow("frame",colour_image)
    cv2.waitKey()
