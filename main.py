import os
import random

import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import cvlib


# Get all Image address, shuffle them and store them in img_files
img_files = [f for f in glob.glob(r'C:\archive\data' + '/**/*', recursive=True)if not os.path.isdir(f)]
random.shuffle(img_files)

im_vector = []
labels = []

# Hyper parameters
rate = 0.001
epoch = 100
batch_size = 64
img_dims = (128, 128, 3)

c = 0
# Convert images to image vector by processing it first to 120 X 120 X 3 and load it in im_vector
for img in img_files:
    try:
        image = cv2.imread(img)
        image_face, confidence = cvlib.detect_face(image)
        face_crop = np.copy(image[image_face[0][1]:image_face[0][3], image_face[0][0]:image_face[0][2]])
        image = cv2.resize(face_crop, (img_dims[0], img_dims[1]))
        image = img_to_array(image)
        im_vector.append(image)

        # Retrieve labels and store it in labels array
        label = img.split(os.path.sep)[-2]
        if label == "women":
            labels.append([1])
        else:
            labels.append([0])

    except Exception as e:
        c = c + 1
    # image.shape()

# Normalize pixel values
print(c)
im_vector = np.array(im_vector, dtype="float")/255.0
labels = np.array(labels)


# data ready to go: split into train and test
trainX, testX, trainY, testY = train_test_split(im_vector, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Generate more pics from same pics like cropped, rotated, etc
extraPics = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                               height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                               horizontal_flip=True, fill_mode="nearest")


def build(width, height, channels, classes):
    model = Sequential()
    inputShape = (height, width, channels)
    channelDim = -1
    # channels are first or last
    if K.image_data_format() == "channels_first":
        inputShape = (channels, height, width)
        channelDim = 1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.50))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model


# build model
model = build(width=img_dims[0], height=img_dims[1], channels=img_dims[2], classes=2)

# compile
optimize = Adam(lr=rate, decay=rate/epoch)
model.compile(loss="binary_crossentropy", optimizer=optimize, metrics=["accuracy"])

# Train
H = model.fit_generator(extraPics.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epoch, verbose=1)



# save model
model.save('male-female-detection.model')

# plot data
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="validation_accuracy")


plt.title("Training Graphs")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig('plt.png')
