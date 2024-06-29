# How to run?: python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
from cvzone.HandTrackingModule import HandDetector

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green



detector = HandDetector(maxHands=2, 
                        detectionCon=0.8) 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('~/output.avi', fourcc, 20.0, (1280, 720))

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# warm up the camera for a couple of seconds
time.sleep(2.0)

# detector = HandDetector(maxHands=1, detectionCon=0.8) 

# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options,
#                                        num_hands=2)
# detector = vision.HandLandmarker.create_from_options(options)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# FPS: used to compute the (approximate) frames per second
# Start the FPS timer
fps = FPS().start()

while True:

    frame = vs.read()
    if frame is None:
        break

    print(frame.shape)
    # detection_result = detector.detect(image) 
    hands, img = detector.findHands(frame)

    if hands:
        finger_count=0
        # For each detected hand
        for hand in hands:
            finger_count =finger_count + sum(detector.fingersUp(hand))

            # Display the finger count on the frame
        cv2.putText(img, f'Palce: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

	# Press 'q' key to break the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()

# In case you removed the opaque tape over your laptop cam, make sure you put them back on once finished ;)
# YAYYYYYYYYYY WE ARE DONE!