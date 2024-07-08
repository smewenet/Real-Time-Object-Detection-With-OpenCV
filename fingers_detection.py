# How to run?: python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import packages

from imutils.video import FPS
import time
import cv2
from cvzone.HandTrackingModule import HandDetector

import imageio

detector = HandDetector(maxHands=2, detectionCon=0.8) 
# initialize the video stream
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
# warm up the camera for a couple of seconds
time.sleep(2.0)

fps = float(vs.get(cv2.CAP_PROP_FPS))
frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Example parameters
output_filename = './output.mp4'
frame_size = (frame_width, frame_height)  # Width, height

# Create a VideoWriter instance
writer = imageio.get_writer(output_filename, fps=fps/2)


try:
    while True:

        ret, frame = vs.read()
        if not ret:
            break

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
        #out_video.write(frame)
        writer.append_data(frame)

        key = cv2.waitKey(10) & 0xFF

        # Press 'q' key to break the loop
        if key == ord('q') or key==27:
            break
except KeyboardInterrupt:
    # Destroy windows and cleanup
    cv2.destroyAllWindows()
    # Stop the video stream
    vs.release()
    writer.close()