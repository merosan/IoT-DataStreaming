#!/usr/bin/python3

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from sense_hat import SenseHat
import face_recognition
import imutils
import pickle
import time
import cv2
import requests  # to send data to Node-RED
from datetime import datetime
import sys  # to exit the script
import json
import os

sense = SenseHat()

# File to store the login state and room counters
state_file = "/home/pi/facial_recognition/state.json"
log_file = "/home/pi/facial_recognition/log.json"
image_dir = "/home/pi/facial_recognition/images/"
climate_log_file = "/home/pi/facial_recognition/climate_log.json"
room_count_file = "/home/pi/facial_recognition/room_count.json"

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "/home/pi/facial_recognition/encodings.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Load the state from the state file or initialize it if the file doesn't exist
def load_state():
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        state = {
            "login_state": {name: False for name in data["names"]},
            "room_counters": {room: 0 for room in set(data["rooms"])}
        }
    return state

# Save the state to the state file
def save_state(state):
    with open(state_file, "w") as f:
        json.dump(state, f)

# Save a log entry to the log file
def save_log_entry(entry):
    try:
        with open(log_file, "r") as f:
            log = json.load(f)
    except FileNotFoundError:
        log = []
    log.append(entry)
    with open(log_file, "w") as f:
        json.dump(log, f)

# Update room counters and climate control log
def update_room_counters_and_climate():
    try:
        with open(climate_log_file, "r") as f:
            climate_log = json.load(f)
    except FileNotFoundError:
        climate_log = {room: {"count": 0, "state": "Off"} for room in set(data["rooms"])}

    for room, count in room_counters.items():
        if count > 0:
            climate_log[room] = {"count": count, "state": "On"}
        else:
            climate_log[room] = {"count": count, "state": "Off"}

    with open(climate_log_file, "w") as f:
        json.dump(climate_log, f)

state = load_state()
login_state = state["login_state"]
room_counters = state["room_counters"]

# initialize the video stream and allow the camera sensor to warm up
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

def get_current_datetime_string():
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the date and time as a string
    datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return datetime_string

def capture_image(name, timestamp):
    image_path = os.path.join(image_dir, f"{name}_{timestamp}.jpg")
    frame = vs.read()
    cv2.imwrite(image_path, frame)
    return image_path

def handle_joystick_event(event):
    if event.action == "pressed" and event.direction == "middle":
        print("Doorbell pressed, starting facial recognition")
        start_facial_recognition()

def start_facial_recognition():
    global currentname, login_state, room_counters  # Use the global variables
    start_time = time.time()
    recognized = False

    # start the FPS counter
    fps = FPS().start()

    while True:
        # grab the frame from the threaded video stream and resize it to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"  # if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of votes
                name = max(counts, key=counts.get)

                # If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)

                    room = data["rooms"][data["names"].index(name)]
                    print(room)

                    logintime = get_current_datetime_string()
                    print(logintime)
                    
                    # Capture an image
                    image_path = capture_image(name, logintime.replace(":", "-").replace(" ", "_"))

                    # Toggle login state
                    if login_state[name]:
                        login_state[name] = False
                        room_counters[room] -= 1
                        state = "logged out"
                        sense.show_message("Bye " + currentname)
                    else:
                        login_state[name] = True
                        room_counters[room] += 1
                        state = "logged in"
                        sense.show_message("Hi " + currentname)

                    # Save the updated state
                    save_state({"login_state": login_state, "room_counters": room_counters})

                    # Save log entry
                    log_entry = {"time": logintime, "name": currentname, "room": room, "state": state, "image": image_path}
                    save_log_entry(log_entry)

                    # Update room counters and climate control
                    update_room_counters_and_climate()

                    recognized = True
                    break

            # update the list of names
            names.append(name)

        if recognized:
            break

        # Check if 10 seconds have passed
        if time.time() - start_time > 10:
            logintime = get_current_datetime_string()
            print(logintime)
            sense.show_message("not recognised")
            # Capture an image
            image_path = capture_image("Unknown", logintime.replace(":", "-").replace(" ", "_"))
            # Save unknown data to log
            log_entry = {"time": logintime, "name": "Unknown", "room": "Unknown", "state": "Unknown", "image": image_path}
            save_log_entry(log_entry)
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    vs.stop()
    sys.exit()  # Exit the script completely

print("Press joystick to start facial recognition")
sense.show_message("press")
sense.stick.direction_middle = handle_joystick_event

# Keep the script running to listen for joystick events
while True:
    pass

# stop the video stream
vs.stop()
