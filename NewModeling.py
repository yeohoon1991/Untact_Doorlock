import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 320)
cam.set(4, 240)
face_detector = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')



print("\n Initializing face capture. Look the camera ")

data = open('/home/pi/testsrc/openCV_ID.txt' , 'r')
face_id = data.read()
count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("/home/pi/testsrc/facede/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 200: # Take 30 face sample and stop video
         break
data.close()


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()