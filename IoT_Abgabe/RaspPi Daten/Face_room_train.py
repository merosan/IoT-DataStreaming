#! /usr/bin/python

# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))

# Initialize the list of known encodings, known names, and rooms
knownEncodings = []
knownNames = []
knownRooms = []

# Define the room assignments
room_assignments = {
    "Sandro": "Room1",
    "Sarah": "Room2",
    "Thierry": "Room3"
}

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model="hog")

    # Compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings
    for encoding in encodings:
        # Add each encoding + name + room to our set of known names, encodings, and rooms
        knownEncodings.append(encoding)
        knownNames.append(name)
        knownRooms.append(room_assignments[name])

# Dump the facial encodings + names + rooms to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames, "rooms": knownRooms}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] encoding and room assignment complete.")
