import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

def speak(message):
    """Convert text to speech."""
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

def load_data():
    """Load names and face data from pickle files."""
    with open('data/names.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    return labels, faces

def initialize_video_capture():
    """Initialize video capture and return the video object."""
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Error: Could not open video capture.")
    return video

def check_if_exists(value):
    """Check if a voter has already voted."""
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    print(f"Found existing vote for: {value}")  # Debugging output
                    return True  # Voter has already voted
    except FileNotFoundError:
        print("Votes.csv not found or unable to open.")
    return False  # Voter has not voted yet

def is_eligible_to_vote(value):
    """Check if a voter is eligible to vote by checking eligible.csv."""
    try:
        with open("eligible.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    print(f"Found eligible entry for: {value}")  # Debugging output
                    return True  # Voter is eligible to vote
    except FileNotFoundError:
        print("eligible.csv not found or unable to open.")
    return False  # Voter is not eligible to vote

def record_vote(output, vote):
    """Record a vote in the CSV file."""
    speak("YOUR VOTE HAS BEEN RECORDED")
    time.sleep(1)
    
    timestamp = datetime.now()
    date_str = timestamp.strftime("%d-%m-%Y")
    time_str = timestamp.strftime("%H:%M:%S")
    
    with open("Votes.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not os.path.isfile("Votes.csv"):
            writer.writerow(['ROLL NO.', 'VOTE', 'DATE', 'TIME'])  # Write header if file is new
        writer.writerow([output[0], vote, date_str, time_str])
    
    speak("THANK YOU FOR VOTING")

def main():
    print("Starting the voting system...")
    
    # Load labels and faces data
    LABELS, FACES = load_data()
    
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    # Initialize video capture
    video = initialize_video_capture()

    # Load Haar Cascade for face detection
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Flag to track if the user has already voted
    already_voted = False

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture video.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        output = None
        
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            
            try:
                output = knn.predict(resized_img)
            except Exception as e:
                print(f"Prediction error: {e}")
                output = None

            # Check if the face belongs to known data
            if output is not None and output[0] in LABELS:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 255), 1)

                # Check if the voter has already voted
                voter_exist = check_if_exists(output[0])
                if voter_exist and not already_voted:
                    print(f"YOU HAVE ALREADY VOTED: {output[0]}")  # Debugging output
                    speak("YOU HAVE ALREADY VOTED")
                    already_voted = True  # Set flag to True to prevent further messages
                    video.release()
                    cv2.destroyAllWindows()  # Clean up resources
                    return  # Exit from main function
                
                # Check if the user is eligible to vote (in eligible.csv)
                eligible_to_vote = is_eligible_to_vote(output[0])
                if not eligible_to_vote:
                    print(f"YOU ARE NOT ELIGIBLE TO VOTE: {output[0]}")  # Debugging output
                    speak("YOU ARE NOT ELIGIBLE TO VOTE")
                    video.release()
                    cv2.destroyAllWindows()  # Clean up resources
                    return  # Exit from main function

                k = cv2.waitKey(1)

                # Display voting options on the screen
                cv2.putText(frame, "Press '1' for Candidate A", (50, frame.shape[0] - 100), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press '2' for Candidate B", (50, frame.shape[0] - 70), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press '3' for Candidate C", (50, frame.shape[0] - 40), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                # Show the frame with options
                cv2.imshow('Voting System', frame)

                # Voting options based on key presses
                if k == ord('1'):
                    record_vote(output, "Candidate A")
                    break
                elif k == ord('2'):
                    record_vote(output, "Candidate B")
                    break
                elif k == ord('3'):
                    record_vote(output, "Candidate C")
                    break

            else:
                print("YOU DO NOT BELONG TO THIS BRANCH")
                speak("YOU DO NOT BELONG TO THIS BRANCH")
                break

        if output is None or output[0] not in LABELS:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
        input("Press Enter to exit...")  # Keep console open until Enter is pressed
    except Exception as e:
        print(f"An error occurred: {e}")