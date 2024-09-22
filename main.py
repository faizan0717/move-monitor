import cv2
import numpy as np
import PosEstimationModule as pm
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Function to update the frame
def update_frame():
    global count, dir, time_series, cap, training_started

    if not training_started:
        return  # Stop if training has not started

    success, img = cap.read()
    if not success:
        return  # Stop if no frame is captured

    img = cv2.resize(img, (640, 480))  # Resize for the canvas
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (430, 60))
        time_series.append([time.time(), per])

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw Bar
        cv2.rectangle(img, (500, 60), (550, 430), color, 3)
        cv2.rectangle(img, (500, int(bar)), (550, 430), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (503, 25), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        cv2.rectangle(img, (29, 16), (118, 122), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    # Convert the image to RGB and display it on the Tkinter label
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    # Call update_frame again after 10 milliseconds
    if training_started:
        root.after(10, update_frame)

# Function to toggle training
def toggle_training():
    global training_started
    if training_started:
        stop_training()
    else:
        start_training()

# Function to start the training
def start_training():
    global training_started
    training_started = True
    button_toggle.config(text="Stop Training")
    update_frame()  # Start updating frames once the training has started

# Function to stop the training
def stop_training():
    global training_started
    global time_series
    global count
    training_started = False
    button_toggle.config(text="Start Training")
    if count != 0:
        # Extract X and Y values for plotting
        x_values = [point[0] for point in time_series]
        y_values = [point[1] for point in time_series]

        # Plot the points as a line graph
        plt.plot(x_values, y_values)

        # Set labels for the axes
        plt.xlabel('Time (seconds)')
        plt.ylabel('Percentage (%)')

        # Add a title
        plt.title('Training Performance Over Time')

        # Save the plot as an image file
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = "./training_data/" + current_time + "_" + str(count) + '.png'
        plt.savefig(image_path)
        plt.close()  # Close the plot to avoid overlap in future plots
        count = 0
        time_series = []

# Function to view or close the history of saved images
def toggle_history():
    if button_history.config('text')[-1] == "View History":
        # Hide the Start Training button and show Next/Previous buttons
        button_toggle.pack_forget()
        button_next.pack(side="left", padx=10, pady=10)
        button_previous.pack(side="left", padx=10, pady=10)
        view_history()
        button_history.config(text="Close History")
    else:
        # Show the Start Training button and hide Next/Previous buttons
        button_toggle.pack(side="left", padx=10, pady=10)
        button_next.pack_forget()
        button_previous.pack_forget()
        label_video.config(image='')
        label_title.config(text="")
        button_history.config(text="View History")

# Function to view the history of saved images
def view_history():
    global current_image_index, image_paths

    # Load the list of image files in the training_data folder
    image_paths = sorted([f for f in os.listdir("./training_data/") if f.endswith('.png')])

    if image_paths:
        current_image_index = 0
        display_image(image_paths[current_image_index])
    else:
        label_video.config(text="No training history available.")

# Function to display an image from the history
def display_image(image_path):
    img = Image.open("./training_data/" + image_path)
    img = img.resize((640, 480))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    label_video.imgtk = img_tk
    label_video.config(image=img_tk)

    # Update the label with the image title (filename)
    image_path = str(image_path)
    time_text = "Time : " + image_path.split("_")[0]
    total_curl = "Total Curls : " +  image_path.split("_")[1].split(".")[0]
    label_title.config(text=time_text+" | "+total_curl)

# Function to go to the next image in history
def next_image():
    global current_image_index, image_paths
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        display_image(image_paths[current_image_index])

# Function to go to the previous image in history
def previous_image():
    global current_image_index, image_paths
    if current_image_index > 0:
        current_image_index -= 1
        display_image(image_paths[current_image_index])

# Initialize pose detector
detector = pm.poseDetector()
count = 0
dir = 0
time_series = []
training_started = False  # Initially, the training hasn't started
current_image_index = -1  # To track current image in history
image_paths = []  # List to store paths to images

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Create the main window
root = tk.Tk()
root.title("Get Fit")
root.geometry("700x600")

# Create a frame for the video and buttons
frame = ttk.Frame(root)
frame.pack(pady=10)

# Create the video label (will display the OpenCV frames)
label_video = tk.Label(frame)
label_video.pack()

# Label to display the image title
label_title = tk.Label(root, text="", font=("Arial", 12))
label_title.pack()

# Create the Toggle Training button (Start/Stop)
button_toggle = tk.Button(root, text="Start Training", command=toggle_training)
button_toggle.pack(side="left", padx=10, pady=10)

# Create the View History button
button_history = tk.Button(root, text="View History", command=toggle_history)
button_history.pack(side="left", padx=10, pady=10)

# Create Next and Previous buttons for navigation (initially hidden)
button_previous = tk.Button(root, text="Previous", command=previous_image)
button_next = tk.Button(root, text="Next", command=next_image)

# Start the Tkinter event loop
root.mainloop()

# Release video capture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
