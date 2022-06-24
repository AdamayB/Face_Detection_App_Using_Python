import cv2
from random import randrange
#Classifier-Just a detector-it is classifying something as a face
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#.xml file already exists==GitHub


webcam=cv2.VideoCapture(0)
while True:
    successful_frame_read,frame=webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)
    cv2.imshow("Detecting Faces", frame)
    key=cv2.waitKey(1)
    #Press Q to exit
    if key==81 or key==113:
        break

print("Done")

#link to the .xml file https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml