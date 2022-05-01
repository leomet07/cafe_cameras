import cv2 # opencv-contrib-python is needed !!
import json
cv2_version = cv2.__version__


import numpy as np
from multiprocessing import Process, Value

cameras = []

with open("cameras.json", "r") as file:
    cameras = json.load(file)

streams = []

def stream_func(connection_url, run, dimensions):
    cap = cv2.VideoCapture(connection_url)

    prev_frame = None

    motion_history = None

    timestamp = 0

    while cap.isOpened() and run.value:
        _, frame = cap.read()
        timestamp += 1
        if not(dimensions is None):
            if "scale" in dimensions:
                xscale = float(dimensions["scale"]["x"])
                yscale = float(dimensions["scale"]["x"])

                resizewidth = int(frame.shape[1] * xscale)
                resizeheight = int(frame.shape[0] * yscale)
                frame = cv2.resize(frame, (resizewidth, resizeheight))


            top_left_point = dimensions["top_left_point"]

            height = dimensions["height"]
            width = dimensions["width"]
            max_height, max_width, _ = frame.shape
            if height == 0:
                height = max_height
            if width == 0:
                width = max_width

            frame = frame[top_left_point["y"]:top_left_point["y"] + height, top_left_point["x"]:top_left_point["x"] + width]

            if motion_history is None:
                motion_history = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

        if prev_frame is None:
            prev_frame = frame
            continue

        diff = cv2.absdiff(frame, prev_frame) # difference between frame and current frame
        
        
        gray= cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # make difference greyscale
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # converting grayscale difference to GaussianBlur (easier to find change)
        
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY) # if pixel value is greater than val, it is assigned white(255) otherwise black
        dilated = cv2.dilate(thresh, None, iterations=4)

        cv2.motempl.updateMotionHistory(dilated, motion_history, timestamp, 10) # 10 is the max history

        motion_countours = motion_history.astype(np.uint8)

        
        to_detect_contours = motion_countours

        # finding contours of moving object
        if str(cv2_version).startswith("3"):
            _, contours, hirarchy = cv2.findContours(
                to_detect_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        else: # For version 4
            contours, hirarchy = cv2.findContours(
                to_detect_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

        
        display_frame = frame.copy()
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 3000  : # or cv2.contourArea(contour) > 10000
                continue
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 20), 2)

        
        motion_history_img = cv2.merge((motion_countours, motion_countours, motion_countours))
        both = np.concatenate((display_frame, motion_history_img), axis=0)

        both = cv2.resize(both, (0, 0), fx=0.7, fy=0.7)

        prev_frame = frame.copy()

        cv2.imshow("Both", both)
        # cv2.imshow("Motion History", motion_history_img)
        # cv2.imshow("Feed", display_frame)

        key = cv2.waitKey(1)

        if key == ord("q") or key == 27:
            run.value = False   


    cap.release()

    cv2.destroyAllWindows()

    


if __name__ == "__main__":
    run = Value('b', True) # Global variable for controlling if the program is running

    for camera in cameras:
        if "use" in camera: 
            use_bool = bool(camera["use"]) 
            if not(use_bool): # Only if the use flag is explicitly set to false is the camera ignored
                continue

        url = camera["url"]
        dimensions =  camera["dimensions"] if "dimensions" in camera else None
        streams.append({ "process" : Process(target=stream_func, args=(url, run, dimensions)), "url" : url} ) # Create processes

    for stream in streams:
        stream["process"].start() # Start all processes

    for stream in streams:
        stream["process"].join() # Wait for all processes to finish before terminating program
        # print("Stream with url of " + stream["url"] + " finished.")

