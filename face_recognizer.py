import cv2
import numpy as np
import FaceDetect as fd
faces_loaded, labels_loaded = fd.BaseLoad("test_s4.csv")
faces = []
labels = []

for x in range(len(faces_loaded)):
    if type(fd.FaceDetect(faces_loaded[x])) != type(None):
        print(type(fd.FaceDetect(faces_loaded[x])))
        faces.append(fd.FaceDetect(faces_loaded[x]))
        labels.append(int(labels_loaded[x], base=15))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.write("reco_4.xml")