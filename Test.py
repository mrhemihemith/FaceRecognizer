import json
import PIL
from PIL import Image
import cv2
import os
import numpy as np
from PIL import Image
import pickle


DATASET_PATH = "Sample Faces"
JSON_PATH = 'test.json'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()

data = {
        "mappings": [],
        "Name": [],
        "files": [],
        "Pixels": []
    }

for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
    # ensure that we are not at the root level
    if dirpath is not DATASET_PATH:
        # update mapping
        category = dirpath.split("/")[-1]  # dataset/down => [dataset,down]
        data["mappings"].append(category)
        print(f"Processing{category}")

        # loop through all the filenames and extract MFCCs
        for f in filenames:
            # get file path
            file_path = os.path.join(dirpath, f)
            print(file_path)

            image = PIL.Image.open(file_path)
            size = image.size
            #print(image)
            #print(size)
            pil_image = Image.open(file_path).convert("L")
            #print(pil_image)
            final_img = pil_image.resize(size, Image.ANTIALIAS)
            #print(final_img)
            #img = cv2.imread(pil_image)
            #cv2.imshow("fil",img)
            #cv2.waitKey()
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array)
            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]

                data["Name"].append(i - 1)  # because dataset =(i=0)#
                data["Pixels"].append(roi.tolist())
                data["files"].append(file_path)
                print(f"{file_path} : {i - 1}")

    # STORE IN A JSON FILE
    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)
