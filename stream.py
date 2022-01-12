from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


model = load_model('male-female-detection.model')

webcam = cv2.VideoCapture(0)
classes = ['men', 'women']

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_mesh:

    while webcam.isOpened():

        status, frame = webcam.read()
        face, confidence = cvlib.detect_face(frame)

        for idx, f in enumerate(face):

            # detected face's coordinates are Lower-left: f[0], f[1], Above-right v: f[2], f[3]
            cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), (0, 255, 0), 2)

            face_crop = np.copy(frame[f[1]:f[3], f[0]:f[2]])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue
            # Set image properties same as model's properties

            face_crop = cv2.resize(face_crop, (128, 128))
            face_crop = face_crop.astype("float")/255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # PREDICTTTTTTTT
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = f[2] - 10 if f[2] - 10 > 10 else f[2] + 10
            cv2.putText(frame, label, (f[0], Y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.7, (0, 255, 0), 2)

            # display output
            cv2.imshow("Male Female detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        #
webcam.release()
cv2.destroyAllWindows()


