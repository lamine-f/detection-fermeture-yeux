import cv2
import os
import time

# Specify the path to the alert sound file
sound_file = 'sounds/beep.mp3'  # Replace with the actual path to your sound file



eye_cascPath = './haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = './haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

"""

$TOLERENCE: parametre 

"""

TOLERENCE = 10

arr = []
count = 0
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()

    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Pour detecter le visage
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            # Detect eyes in the face
            frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            count=count+1

            #Si l'oeil est ouvert
            if len(eyes) == 0:
                arr.append(0)
                #print('no eyes!!!')
                #print("oeil fermé")

            #Si l'oeil est fermé
            else:
                arr.append(1)
                #print('eyes!!!')
                #print("oeil ouvert")

            #print(count)
            if ( count%TOLERENCE == 0 and sum(arr[count-TOLERENCE:count]) == 0 ):
                print("Vous vous endormez!!")
                # Play the alert sound
                os.system(f'paplay {sound_file}')
                time.sleep(1)  # Pause for 1 second after playing the sound
                #beepy.beep("coin")
