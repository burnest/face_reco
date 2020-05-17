import cv2
import FaceDetect as fd
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

test_img = cv2.imread("lena.png", 1)
test2_img = cv2.imread("ExtendedYaleB/s4/yaleB11/yaleB11_P00A+000E+20.pgm",1)
test3_img = cv2.imread("ExtendedYaleB/s4/yaleB13/yaleB13_P00A+000E+90.pgm",1)
test4_img = cv2.imread("ExtendedYaleB/s4/yaleB19/yaleB19_P00A+000E-20.pgm",1)
test5_img = cv2.imread("ExtendedYaleB/s4/yaleB11_P00A+015E+20.pgm",1)
test6_img = cv2.imread("ExtendedYaleB/s4/yaleB19_P00A-020E-40.pgm",1)
test7_img = cv2.imread("ExtendedYaleB/s4/yaleB13_P00A-005E+10.pgm",1)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("recognizers/reco_4.xml")

# make a copy of the image as we don't want to change original image
img = test_img.copy()
img2 = test2_img.copy()
img3 = test3_img.copy()
img4 = test4_img.copy()
img5 = test5_img.copy()
img6 = test6_img.copy()
img7 = test7_img.copy()
# detect face from the image
face = fd.FaceDetect(img)
face2 = fd.FaceDetect(img2)
face3 = fd.FaceDetect(img3)
face4 = fd.FaceDetect(img4)
face5 = fd.FaceDetect(img5)
face6 = fd.FaceDetect(img6)
face7 = fd.FaceDetect(img7)
# predict the image using our face recognizer

print("image from base")
label2 = face_recognizer.predict(face2)
print("to jest warostsc predykcji dla yaleb11")
print(label2)

label3 = face_recognizer.predict(face3)
print("to jest warostsc predykcji dla yaleb13")
print(label3)

label4 = face_recognizer.predict(face4)
print("to jest warostsc predykcji dla yaleb19")
print(label4)

print("image outside of base")

label5 = face_recognizer.predict(face5)
print("to jest warostsc predykcji dla yaleb11")
print(label5)

label6 = face_recognizer.predict(face6)
print("to jest warostsc predykcji dla yaleb13")
print(label6)

label7 = face_recognizer.predict(face7)
print("to jest warostsc predykcji dla yaleb19")
print(label7)

label = face_recognizer.predict(face)
print("to jest warostsc predykcji dla lena")
print(label)



