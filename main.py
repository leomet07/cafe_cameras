import cv2 # opencv-contrib-python is needed !!
import json
cv2_version = cv2.__version__


import numpy as np
from multiprocessing import Process, Value

cameras = []

with open("cameras.json", "r") as file:
    cameras = json.load(file)

streams = []

def stream_func(connection_url, run, dimensions, index):
    cap = cv2.VideoCapture(connection_url)

    prev_frame = None

    motion_history = None

    videowritier = None

    timestamp = 0
    cooldowns = []

    while cap.isOpened() and run.value:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of stream.")
            break
        # Time stamp incrementation
        timestamp += 1 # Global timestamp

        cooldowns_to_remove = [] # By index
        for cooldown_i in range(0, len(cooldowns)):
            cooldowns[cooldown_i]["elapsed_cooldown_frames"] += 1 # Increment elapsed counter
            # If cooldown has finally been reached
            if cooldowns[cooldown_i]["elapsed_cooldown_frames"] > cooldowns[cooldown_i]["goal_cooldown_frames"]:
                # then remove this entry later
                cooldowns_to_remove.append(cooldown_i)
        # Remove AFTER for loop as to keep rest of list/list order intact and not break loop
        for remove_index in cooldowns_to_remove:
            del cooldowns[remove_index] 
        
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

            if videowritier is None:
                videowritier = cv2.VideoWriter("output_" + str(index) + ".avi" ,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

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

        wait_val = 1
        display_frame = frame.copy()
        for contour in contours:
            (x1, y1, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 3000 or cv2.contourArea(contour) > 200000 : 
                continue
            x2 = x1 + w
            y2 = y1 + h

            mx = round((x1 + x2) / 2)
            mframe = round(frame.shape[1] / 2)

            if (abs(mframe - mx) <= 10):
                # wait_val = 0 # To pause the program
                
                # Check if it overlaps with something already on the cooldown
                overlap = False
                for cooldown in cooldowns: 
                    cooldown_coordinates = cooldown["coordinates"]
                    cy1 = cooldown_coordinates["y1"]
                    cy2 = cooldown_coordinates["y2"]
                    
                    # IF the line segments overlap
                    if ((cy1 <= y1 and cy2 >= y1) or (cy2 >= y2 and cy1 <= y2)):
                        overlap = True
                        break # out of inner for loop
                

                if overlap:
                    break # Not valid to count, so exit
                print("In middle")
                # reset motion history
                motion_history = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
                # , however next frame's motion will still detect a difference, so we flag next frame
                cooldowns.append({
                    "elapsed_cooldown_frames" : 0,
                    "goal_cooldown_frames" : 20, # Hard coded cooldown time
                    "coordinates" : {
                        "x1" : x1,
                        "y1" : y1,
                        "x2" : x2,
                        "y2" : y2,
                    }
                })
                
                print(mx, mframe, "Timestamp: ", timestamp)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 20), 2)

        
        motion_history_img = cv2.merge((motion_countours, motion_countours, motion_countours))
        both = np.concatenate((display_frame, motion_history_img), axis=0)

        both = cv2.resize(both, (0, 0), fx=0.7, fy=0.7)

        prev_frame = frame.copy()

        cv2.imshow("Both", both)
        
        # cv2.imshow("Motion History", motion_history_img)
        # cv2.imshow("Feed", display_frame)

        key = cv2.waitKey(wait_val)

        videowritier.write(frame)

        if key == ord("q") or key == 27:
            run.value = False   


    cap.release()

    cv2.destroyAllWindows()

    


if __name__ == "__main__":
    run = Value('b', True) # Global variable for controlling if the program is running

    for index in range(len(cameras)):
        camera = cameras[index]
        if "use" in camera: 
            use_bool = bool(camera["use"]) 
            if not(use_bool): # Only if the use flag is explicitly set to false is the camera ignored
                continue

        url = camera["url"]
        dimensions =  camera["dimensions"] if "dimensions" in camera else None
        streams.append({ "process" : Process(target=stream_func, args=(url, run, dimensions, index)), "url" : url} ) # Create processes

    for stream in streams:
        stream["process"].start() # Start all processes

    for stream in streams:
        stream["process"].join() # Wait for all processes to finish before terminating program
        # print("Stream with url of " + stream["url"] + " finished.")

