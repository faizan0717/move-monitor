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
import sqlite3
import hashlib

# -------------------- DATABASE FUNCTIONS --------------------

def initialize_database():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT
    )
    ''')
    
    # Create training_history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        exercise TEXT NOT NULL,
        count INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        image_path TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash the password for security using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=None):
    """Register a new user."""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    
    hashed_password = hash_password(password)
    try:
        cursor.execute('''
        INSERT INTO users (username, password, email) VALUES (?, ?, ?)
        ''', (username, hashed_password, email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Username already exists

def login_user(username, password):
    """Login the user with the username and password."""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    hashed_password = hash_password(password)
    cursor.execute('''
    SELECT * FROM users WHERE username=? AND password=?
    ''', (username, hashed_password))
    
    user = cursor.fetchone()
    conn.close()

    return user  # Returns None if no user found, otherwise user data

def save_training_history(user_id, exercise, count, image_path):
    """Save training history for the user."""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO training_history (user_id, exercise, count, timestamp, image_path)
    VALUES (?, ?, ?, ?, ?)
    ''', (user_id, exercise, count, timestamp, image_path))
    
    conn.commit()
    conn.close()

def get_training_history(user_id):
    """Retrieve training history for the user."""
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM training_history WHERE user_id=?
    ''', (user_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    return history

# -------------------- EXERCISE TRACKING FUNCTIONS --------------------

# Initialize pose detector
detector = pm.poseDetector()
count = 0
dir = 0
time_series = []
training_started = False  # Initially, the training hasn't started
current_image_index = -1  # To track current image in history
image_paths = []  # List to store paths to images
current_user = None  # To store the currently logged-in user

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# -------------------- EXERCISE LOGIC --------------------

def update_frame():
    global count, dir, time_series, cap, training_started, current_exercise

    if not training_started or not current_user:
        return  # Stop if training has not started or user is not logged in

    success, img = cap.read()
    if not success:
        return  # Stop if no frame is captured

    img = cv2.resize(img, (640, 480))  # Resize for the canvas
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img)

    per = 0  # Initialize 'per' to ensure it's always defined

    if len(lmList) != 0:
        if current_exercise.get() == "Dumbbell Curls":
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (430, 60))
            check_curls(per)
        elif current_exercise.get() == "Squats":
            angle = detector.findAngle(img, 11, 12, 14)  # Adjust for squat angles
            per = np.interp(angle, (190, 310), (0, 100))
            bar = np.interp(angle, (210, 310), (430, 60))
            check_squats(per)
        elif current_exercise.get() == "Push-ups":
            angle = detector.findAngle(img, 11, 13, 15)  # Adjust for push-up angles
            per = np.interp(angle, (210, 330), (0, 100))
            bar = np.interp(angle, (220, 330), (430, 60))
            check_pushups(per)

        time_series.append([time.time(), per])  # Now 'per' is guaranteed to be defined

        # Draw Bar
        draw_progress_bar(img, bar, per)

    # Convert the image to RGB and display it on the Tkinter label
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    # Call update_frame again after 10 milliseconds
    if training_started:
        root.after(10, update_frame)

def check_curls(per):
    global count, dir
    if per == 100:
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        if dir == 1:
            count += 0.5
            dir = 0

def check_squats(per):
    global count, dir
    if per == 100:
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        if dir == 1:
            count += 0.5
            dir = 0

def check_pushups(per):
    global count, dir
    if per == 100:
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        if dir == 1:
            count += 0.5
            dir = 0

def draw_progress_bar(img, bar, per):
    color = (0, 255, 255)  # Default color
    cv2.rectangle(img, (500, 60), (550, 430), color, 3)
    cv2.rectangle(img, (500, int(bar)), (550, 430), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (503, 25), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.rectangle(img, (29, 16), (118, 122), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

# -------------------- TRAINING CONTROLS --------------------

def toggle_training():
    global training_started
    if training_started:
        stop_training()
    else:
        start_training()

def start_training():
    global training_started
    if not current_user:
        please_login_status.config(text="Please log in to start training.", fg="red")
        return
    training_started = True
    button_toggle.config(text="Stop Training")
    update_frame()

def stop_training():
    global training_started, time_series, count
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
        plt.title(f'{current_exercise.get()} Training Performance Over Time')

        # Save the plot as an image file
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = f"./training_data/user_{current_user[0]}_{current_time}_{count}.png"
        plt.savefig(image_path)
        plt.close()  # Close the plot to avoid overlap in future plots

        # Save training history to the database
        save_training_history(current_user[0], current_exercise.get(), count, image_path)

        count = 0
        time_series = []

# -------------------- HISTORY FUNCTIONS --------------------

def view_history():
    global current_image_index, image_paths

    if not current_user:
        history_login_status.config(text="Please log in to view history.", fg="red")
        return

    history = get_training_history(current_user[0])
    if history:
        image_paths = [record[5] for record in history]  # Extract image paths
        current_image_index = 0
        display_image(image_paths[current_image_index])
    else:
        label_video.config(text="No training history available.")

# def display_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((640, 480))  # Resize for display
#     img_tk = ImageTk.PhotoImage(img)
#     label_video.imgtk = img_tk
#     label_video.config(image=img_tk)

def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((640, 480))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    label_video_history.imgtk = img_tk
    label_video_history.config(image=img_tk)

    # Extract time and total curls from the image path
    image_path = str(image_path)
    time_text = "Time: " + image_path.split("_")[2] + " " + image_path.split("_")[3].split(".")[0]
    total_curl = "Total Curls: " + image_path.split("_")[4].split(".")[0]

    # Update the label below the graph
    label_history_info.config(text=time_text + " | " + total_curl)

    
def next_image():
    global current_image_index, image_paths
    if current_image_index + 1 < len(image_paths):
        current_image_index += 1
        display_image(image_paths[current_image_index])

def previous_image():
    global current_image_index, image_paths
    if current_image_index - 1 >= 0:
        current_image_index -= 1
        display_image(image_paths[current_image_index])

# -------------------- GUI --------------------

root = tk.Tk()
root.title("Fitness Tracking App")
root.geometry("700x600")

# Initialize database
initialize_database()

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Create pages
pages = {}
login_page = ttk.Frame(notebook)
register_page = ttk.Frame(notebook)
training_page = ttk.Frame(notebook)
history_page = ttk.Frame(notebook)

# Add pages to the notebook
notebook.add(login_page, text="Login")
notebook.add(register_page, text="Register")
notebook.add(training_page, text="Training")
notebook.add(history_page, text="History")

# Store page references
pages["login"] = login_page
pages["register"] = register_page
pages["training"] = training_page
pages["history"] = history_page

# -------------------- LOGIN PAGE --------------------

tk.Label(login_page, text="Username:").pack(pady=5)
username_entry = tk.Entry(login_page)
username_entry.pack(pady=5)

tk.Label(login_page, text="Password:").pack(pady=5)
password_entry = tk.Entry(login_page, show="*")
password_entry.pack(pady=5)

login_status = tk.Label(login_page, text="")
login_status.pack(pady=5)

please_login_status = tk.Label(training_page, text="")
please_login_status.pack(pady=5)

history_login_status= tk.Label(history_page, text="")
history_login_status.pack(pady=5)

def handle_login():
    username = username_entry.get()
    password = password_entry.get()

    user = login_user(username, password)
    if user:
        global current_user
        current_user = user
        please_login_status.config(text="", fg="red")
        history_login_status.config(text="", fg="red")
        label_title.config(text=f"Welcome, {user[1]}!")
        notebook.select(training_page)
    else:
        login_status.config(text="Invalid credentials, please try again.", fg="red")

tk.Button(login_page, text="Login", command=handle_login).pack(pady=10)
tk.Button(login_page, text="Go to Register", command=lambda: notebook.select(register_page)).pack(pady=10)

# -------------------- REGISTER PAGE --------------------

tk.Label(register_page, text="Username:").pack(pady=5)
register_username_entry = tk.Entry(register_page)
register_username_entry.pack(pady=5)

tk.Label(register_page, text="Password:").pack(pady=5)
register_password_entry = tk.Entry(register_page, show="*")
register_password_entry.pack(pady=5)

tk.Label(register_page, text="Email (optional):").pack(pady=5)
email_entry = tk.Entry(register_page)
email_entry.pack(pady=5)

register_status = tk.Label(register_page, text="")
register_status.pack(pady=5)

def handle_register():
    username = register_username_entry.get()
    password = register_password_entry.get()
    email = email_entry.get()

    if username and password:
        if register_user(username, password, email):
            register_status.config(text="Registration successful! You can now login.", fg="green")
        else:
            register_status.config(text="Username already exists.", fg="red")

tk.Button(register_page, text="Register", command=handle_register).pack(pady=10)
tk.Button(register_page, text="Go to Login", command=lambda: notebook.select(login_page)).pack(pady=10)

# -------------------- TRAINING PAGE --------------------

label_title = tk.Label(training_page, text="Welcome to Fitness Tracking!", font=("Arial", 16))
label_title.pack(pady=10)

label_video = tk.Label(training_page)
label_video.pack(pady=10)

current_exercise = tk.StringVar(value="Dumbbell Curls")
exercise_menu = ttk.Combobox(training_page, textvariable=current_exercise, values=["Dumbbell Curls", "Squats", "Push-ups"])
exercise_menu.pack(pady=10)

button_toggle = tk.Button(training_page, text="Start Training", command=toggle_training)
button_toggle.pack(pady=10)

tk.Button(history_page, text="View History", command=view_history).pack(pady=10)

# -------------------- HISTORY PAGE --------------------

tk.Button(history_page, text="Previous", command=previous_image).pack(pady=10)
tk.Button(history_page, text="Next", command=next_image).pack(pady=10)

label_video_history = tk.Label(history_page)
label_video_history.pack(pady=10)

# Add this line to the History Page section:
label_history_info = tk.Label(history_page, text="", font=("Arial", 12))
label_history_info.pack(pady=10)

# Show the login page by default
notebook.select(login_page)

root.mainloop()