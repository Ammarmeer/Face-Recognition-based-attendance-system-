from email.mime import image
from unicodedata import name
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#creating a path of the folder in which our images are saved
#here our images are saved in images folder
#so we have given the path name images (images as our dataset)
#path = 'images'

path='images'
#creating a list
images=[] #list of images
personName=[]
#list in wich the path is saved (path of images)
myDataList=os.listdir(path)
print(myDataList)
#now we extract the names of images from the list
#for loop to access the list of images
#using open cv to read all the images
#_____________#
for cu_img in myDataList:
    current_img=cv2.imread(f'{path}/{cu_img}')
    #adding the read images to the image list
    images.append(current_img)
    #to get names of the images to the list
    #splitting the text
    #accessing the zero'th element of the cu img which is name
    personName.append(os.path.splitext(cu_img)[0])
print(personName)
def faceEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#bgr ko rgb mein convert krna
        encode=face_recognition.face_encodings(img)[0]#images ka 1st element lena sath encoding kr rhy hain hum ye
        encodeList.append(encode)#encode list mein add kr dena sab
    return encodeList


    
encodeListKnown=(faceEncodings(images))
print("All encodings Complete.......")
#this was hog algorithm _____#

#function to mark attendance
def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')




cap=cv2.VideoCapture(0)

while True:
    ret,frame= cap.read()
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

    facesCurrentFrame=face_recognition.face_locations(faces)
    encodesCurrentFrame=face_recognition.face_encodings(faces,facesCurrentFrame)
    
    for encodeFace, faceLoctn in zip(encodesCurrentFrame,facesCurrentFrame):
        matches=face_recognition.compare_faces(encodeListKnown, encodeFace)
        #to find the distance between faces the minimum more matched
        faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace)

        #to find minimum distance from the facedistance
        matchIndex=np.argmin(faceDistance)
        if matches[matchIndex]:
            nameOfPerson=personName[matchIndex].upper()
            #print(nameOfPerson)
            y1,x2,y2,x1=faceLoctn
            y1,x2,y2,x1 = y1*4, x2*4 , y2*4 , x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,nameOfPerson,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            attendance(nameOfPerson)
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(10)==13:
        break
cap.release()
cv2.destroyAllWindows()






