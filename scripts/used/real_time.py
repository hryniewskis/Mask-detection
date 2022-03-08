import os
import cv2
import keras
import numpy as np

path = os.getcwd() + "/scripts/imagenet.h5"
imagenet = keras.models.load_model("scripts/imagenet.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)

while True:

    # Capture the video frame
    # by frame
    ret, im = vid.read()
    #print(im)
    # Display the resulting frame
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    else:
        #iterater trough faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = im[y:y + h, x:x + w]
            rerect_sized = cv2.resize(face_img, (224, 224))
            normalized = rerect_sized / 255.0
            reshaped = np.reshape(normalized, (1, 224, 224, 3))
            reshaped = np.vstack([reshaped])
            result = imagenet.predict(reshaped)
            print(result)

            if result[0][0] > 0.5:
                cv2.putText(im, "No mask!", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if result[0][0] <= 0.5:
                cv2.putText(im, "Mask" + str(), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,128,0),2)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 128, 0), 2)


    cv2.imshow('frame', im)
    #print(result)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()