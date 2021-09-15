import cv2
import numpy as np
from PIL import Image
import os
import random

# Path for face image database
dataset = '/home/pi/testsrc/facede/dataset'
path = dataset
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

os.system('sudo rm /home/pi/testsrc/facede/trainer/*')
print(" remove image done ")
def getImagesAndLabels(path):
    randomint = random.randint(1,10000)
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        # id = int(os.path.split(imagePath)[-1].split(".")[1])
        id = randomint 
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples , ids
print ("\n Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))


recognizer.write('/home/pi/testsrc/facede/trainer/trainer.yml')
print("\n faces trained")
#print("\n [INFO] {0} faces trained. Exiting Program")



recognizer.read('/home/pi/testsrc/facede/trainer/trainer.yml')
cascadePath = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 1

names = ['None', 'Known']

cam = cv2.VideoCapture(0)
cam.set(3, 320) # set video widht
cam.set(4, 240) # set video height

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

kncnt = 1
uncnt = 1
kk = True
while kk:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = "known"
            confidence = "  {0}%".format(round(100 - confidence))
            kncnt+=1 
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            uncnt+=1
            
            
        if kncnt == 100:
            f = open("/home/pi/testsrc/openCV_confidence.txt",'w')
            f.write("1")
            f.close()
            print("/n Open door ")
            kk = False
            break
        if uncnt == 400:
            f2 = open("/home/pi/testsrc/openCV_confidence.txt",'w')
            f2.write("0")
            f2.close()
            print("/n Cant open door ")
            kk = False
            break
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        

    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n Exiting Program ")
cam.release()
exit()