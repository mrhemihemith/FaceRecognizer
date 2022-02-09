import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "sample_face_gray.json"
SAVED_MODEL_PATH = "model_sam_gray.h5"
LEARNING_RATE = 0.0001
EPOCHS = 5
BATCH_SIZE = 2
NUM_PERSONS = 3




def load_dataset(Data_path):
    with open(Data_path,"r") as fp:
        data = json.load(fp)

        X = data["Pixels"]
        Y = data["Name"]
        return X,Y

def build_model(input_shape,learning_rate,error="sparse_categorical_crossentropy"):

    #build network
    model = keras.Sequential()

    # 1st Cnn layer
    model.add(keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2),strides = (2,2),padding="same"))

    # 2nd Cnn layer
    model.add(keras.layers.Conv2D(16, (3, 3), activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2), strides=(2, 2), padding="same"))

    # 3rd Cnn layer
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))


    # flatten the output and feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))#reducing dependencies on one layer

    # output with softmax
    model.add(keras.layers.Dense(NUM_PERSONS,activation="softmax"))


    #compile network
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=error,
                  metrics=["accuracy"])

    # model summary
    model.summary()

    return model



def get_datasplits(data_path,test_size=0.1,validtaion_size=0.1):
    X , Y = load_dataset(data_path)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validtaion_size)


    return X_train,X_validation,X_test,Y_train,Y_validation,Y_test




def main():

    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = get_datasplits(DATA_PATH)


    input_shape = (X_train.shape[1],X_train.shape[2],X_train[3])
    print(f"input_shape:{input_shape}")
    model = build_model(input_shape=input_shape,learning_rate=LEARNING_RATE)

    model.fit(X_train,Y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_data=(X_validation,Y_validation))

    test_error,test_accuracy = model.evaluate(X_test,Y_test)
    print(f"Test Error:{test_error},Test Accuracy:{test_accuracy}")

    model.save(SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()

