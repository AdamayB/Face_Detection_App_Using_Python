import cv2
from random import randrange
#Classifier-Just a detector-it is classifying something as a face

trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#.xml file already exists==GitHub
s1=input("Enter Full File Path:")

#Reading The Image
img=cv2.imread(s1)

#grey-scaling
grayscale_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detecting Faces
#returns coordinates of rectangles surrounding the face
face_coordinates=trained_face_data.detectMultiScale(grayscale_img)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),2)
#                  ^                 |^|        color   ^thickness of the rectangle
#   coordinates of top-left   cdnts of bottom right
#Showing the image in another window
cv2.imshow("Detecting Faces",img)

#waits for you to press key
cv2.waitKey()

print("Done")



#link to the .xml file https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml