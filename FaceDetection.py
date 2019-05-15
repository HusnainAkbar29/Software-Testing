
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import dlib

import os

#set confidence percentage
conf = 0.3

# loading our CNN model
print("***Loading Model***")
net = cv2.dnn.readNetFromCaffe('Models/face_detection.prototxt.txt','Models/face_detection.caffemodel')
print("***Starting Video Stream***")

class FaceVideo(object):
    def __init__(self):
        self.video = cv2.VideoCapture("faces closed.mp4")


    def __del__(self):
        self.video.release()

    def get_frame(self):
        face_count = 0
        success, frame = self.video.read()

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        #forward propogation of frame in convolutional layers of model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < conf:
                continue

            else: face_count = face_count + 1

        
            bounded_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = bounded_box.astype("int")
        
            conf_value = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
            cv2.putText(frame, conf_value, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        totalfaces = "Total Faces: {:.0f}".format(face_count)
        cv2.putText(frame, totalfaces, (10, h - (1 * 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(frame, totalfaces, (20, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
