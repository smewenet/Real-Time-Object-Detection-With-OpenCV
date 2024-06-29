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

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# warm up the camera for a couple of seconds
time.sleep(2.0)

# detector = HandDetector(maxHands=1, detectionCon=0.8) 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# FPS: used to compute the (approximate) frames per second
# Start the FPS timer
fps = FPS().start()

while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	# vs is the VideoStream
	frame = vs.read()
	cv_mat = cv2.imread(frame)
	image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
	#image = mp.Image(image_format=mp.ImageFormat.GRAY8, data=cv_mat)

	detection_result = detector.detect(image) 

	annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
	cv2.imshow("Video", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	# if hand:
    #       # Taking the landmarks of hand 
	# 	lmlist = hand[0]  
	# 	print(lmlist)
	# 	# if lmlist: 
            
    #           # Find how many fingers are up 
    #         # This function return list 
	# 		# fingerup = detector.fingersUp(lmlist)   
	# 		# print(fingerup)

    # # # Resize the image 
	# # fing = cv2.resize(fing, (220, 280)) 
	# # frame[50:330, 20:240] = fing 
      
 

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