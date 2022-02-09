import json
import os
import cv2
import librosa
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
DATASET_PATH = "Sample Faces"
JSON_PATH  = "sample_face_colour.json"

def prepare_dataset(dataset_path,json_path):
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
                # print(file_path)
                colour_image = cv2.imread(file_path)
                gray_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)

                pil_image = Image.open(file_path).convert("L")
                size = (550, 550)
                final_img = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(pil_image, "uint8")
                #print(image_array)
                faces = face_cascade.detectMultiScale(image_array)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h,x:x+w]
                    x_train = roi
                    #print(x_train)


                    # store data
                    data["Name"].append(i - 1)  # because dataset =(i=0)#
                    data["Pixels"].append(roi.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path} : {i - 1}")
                    #print(n)


    # STORE IN A JSON FILE
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)



if __name__ == '__main__':
    prepare_dataset(DATASET_PATH,JSON_PATH)





'''''    
import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"Sample Faces")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png") or file.endswith("JPG"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label,path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                id_ = label_ids[label]
                print(label_ids)


                pil_image = Image.open(path).convert("L")
                size =(550, 550)
                final_img = pil_image.resize(size, Image.ANTIALIAS)



                image_array = np.array(pil_image,"uint8")
                print(image_array)
                faces = face_cascade.detectMultiScale(image_array)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h,x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

                    print(x_train)
                    print(y_labels)

                    with open("labels.pickle","wb") as f:
                        pickle.dump(label_ids,f)

                    recognizer.train(x_train, np.array(y_labels))
                    recognizer.save("trainner.json")
                    print(roi)


'''''