#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python opencv-contrib-python numpy


# In[3]:


from IPython.display import display, Image


# In[4]:


import os
os.getcwd()


# In[ ]:


import cv2
import os
import numpy as np
def encode_criminal_faces():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    criminal_face_encodings = []
    criminal_face_names = []
    criminal_faces_folder = "C:/Users/saikr/criminal"

    for filename in os.listdir(criminal_faces_folder):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(criminal_faces_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            criminal_face_encodings.append(gray)
            criminal_face_names.append(filename.split('.')[0])
    face_recognizer.train(criminal_face_encodings, np.array(range(len(criminal_face_encodings))))

    return face_recognizer, criminal_face_names
def log_criminal_detection(name):
    with open("criminal_detection_log.txt", "a") as file:
        file.write(f"Criminal {name} detected at {cv2.getTickCount()}\n")
    print(f"Criminal {name} detected!")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer, criminal_face_names = encode_criminal_faces()
cap = cv2.VideoCapture(0)
logged_criminals = []

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Predict the label (name) and confidence (score)
        label, confidence = face_recognizer.predict(face_roi)

        # If the confidence is low, it means the face is recognized
        if confidence < 10:
            name = criminal_face_names[label]

            # Only log if this name hasn't been logged yet
            if name not in logged_criminals:
                log_criminal_detection(name)
                logged_criminals.append(name)
        else:
            name = "Unknown"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Criminal Detection System', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




