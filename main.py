import cv2
from multiprocessing import Process, Value

connection_urls = []

with open("urls.txt", "r") as file:
    for line in file.readlines():
        clean = line.strip()
        if not(clean.startswith("#")):
            connection_urls.append(clean)

streams = []

def stream_func(connection_url, run):
    cap = cv2.VideoCapture(connection_url)

    while cap.isOpened() and run.value:
        _, frame = cap.read()

        cv2.imshow("Feed", frame)

        key = cv2.waitKey(1)

        if key == ord("q") or key == 27:
            run.value = False   


    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run = Value('b', True) # Global variable for controlling if the program is running

    for url in connection_urls:
        streams.append(Process(target=stream_func, args=(url, run ))) # Create processes

    for stream in streams:
        stream.start() # Start all processes

    for stream in streams:
        stream.join() # Wait for all processes to finsih before terminating program

