import cv2
import json
cv2_version = cv2.__version__

import numpy as np
from multiprocessing import Process, Value

cameras = []

with open("cameras.json", "r") as file:
    cameras = json.load(file)

streams = []

def stream_func(connection_url, run):
    cap = cv2.VideoCapture(connection_url)

    prev_frame = None
    while cap.isOpened() and run.value:
        _, frame = cap.read()
        
        if prev_frame is None:
            prev_frame = frame
            continue

        
        diff = cv2.absdiff(prev_frame, frame) # difference between frame and current frame
        
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # make difference greyscale
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # converting grayscale difference to GaussianBlur (easier to find change)
        
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY) # if pixel value is greater than val, it is assigned white(255) otherwise black
        dilated = cv2.dilate(thresh, None, iterations=4)

        # finding contours of moving object
        if str(cv2_version).startswith("3"):
            _, contours, hirarchy = cv2.findContours(
                dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        else: # For version 4
            contours, hirarchy = cv2.findContours(
                dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

        detected = False

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 3000:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 20), 2)
            detected = True

        drawn_thresh = cv2.add(frame, frame, mask=thresh) # Difference frame 
        both = np.concatenate((frame, drawn_thresh), axis=0)

        both = cv2.resize(both, (0, 0), fx=0.7, fy=0.7)

        # cv2.imshow("Feed", both)
        cv2.imshow("Feed", frame)

        key = cv2.waitKey(1)

        if key == ord("q") or key == 27:
            run.value = False   


    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run = Value('b', True) # Global variable for controlling if the program is running

    for camera in cameras:
        url = camera["url"]
        streams.append({ "process" : Process(target=stream_func, args=(url, run )), "url" : url} ) # Create processes

    for stream in streams:
        stream["process"].start() # Start all processes

    for stream in streams:
        stream["process"].join() # Wait for all processes to finish before terminating program
        print("Stream with url of " + stream["url"] + " finished.")

