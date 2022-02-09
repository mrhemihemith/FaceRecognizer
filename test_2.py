import cv2
import os
from PIL import Image
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR,"images")


current_id = 1
label_ids = {}
y_labels = []
x_train = []


for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            if label in  label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id = label_ids[label]
            #print(id)
            #print(path)
            #print(label)
            #y_labels.append(label) #we should keep the labels in an integer form#
            #x_train.append(path) #and the images in an numpy array#
            pil_image = Image.open(path).convert("L") # opens amd converts the image into grayscale#
            #size = (550,550)
            #final_img = pil_image.resize(size,Image.ANTIALIAS)

            image_array = np.array(pil_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            #print(image_array)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id)

#print(x_train)
#print(y_labels)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")     
