Below is a complete and structured version of your face recognition project using Python, OpenCV, and the face_recognition library. This version includes comments for better understanding and a clear structure for the project.



images/: Directory containing images of known faces.
main.py: The main script for face recognition.
requirements.txt: List of required Python packages.


import cv2
import numpy as np
import face_recognition
import os

# Path to the directory containing known faces
path = r"C:\\Users\\joyson paul pinto\\Desktop\\images"

# Lists to store known face encodings and their corresponding names
known_faces = []
known_names = []

# Load known faces and their names
for file in os.listdir(path):
    image = face_recognition.load_image_file(os.path.join(path, file))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    name = os.path.splitext(file)[0]
    known_names.append(name)

# Initialize video capture for the first camera
cap1 = cv2.VideoCapture(0)

# Initialize video capture for the second camera (if needed)
# cap2 = cv2.VideoCapture("http://00.000.0.0000:000000")  # enter your IP address

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\joyson paul pinto\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

# List to store detected names
detected_names = []

while True:
    # Read frames from the cameras
    ret1, frame1 = cap1.read()
    # ret2, frame2 = cap2.read()

    if ret1:
        # Resize the frame for better performance
        frame1 = cv2.resize(frame1, (640, 480))
        # frame2 = cv2.resize(frame2, (640, 480))

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        for (x, y, w, h) in faces:
            face = frame1[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            if len(face_recognition.face_encodings(face_rgb)) > 0:
                face_encoding = face_recognition.face_encodings(face_rgb)[0]
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Intruder"
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]

                if name not in detected_names:
                    detected_names.append(name)
                    print(name)

                # Draw a rectangle around the face and label it with the name
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame1, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frames in separate windows
    cv2.imshow("Camera 1", frame1)
    # cv2.imshow("Camera 2", frame2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close all OpenCV windows
cap1.release()
# cap2.release()
cv2.destroyAllWindows()
