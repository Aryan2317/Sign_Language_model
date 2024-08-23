import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from PIL import Image, ImageTk
import numpy as np
import cv2
import mediapipe as mp
import os

def SignLanguageModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
    return model


xml_path = r'C:\Users\DELL\Downloads\hands.md'
if not os.path.isfile(xml_path):
    raise FileNotFoundError(f"The file {xml_path} does not exist. Please download it from the OpenCV repository.")

top = tk.Tk()
top.geometry('800x600')
top.title('Sign Language Model')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('Arial', 20, 'bold'))
sign_image = Label(top)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=1.0, min_tracking_confidence=0.7)

def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pred = "Recognized Gesture"  
            label1.configure(foreground="#011638", text=pred)
        else:
            label1.configure(foreground="#011638", text="No hands detected")
    except Exception as e:
        print(e)
        label1.configure(foreground="#011638", text="Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Gesture", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(font='bold')
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width() / 2.3, top.winfo_height() / 2.3))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(e)

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('Arial', 12, 'bold'))
upload.pack(side='bottom')
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
heading = Label(top, text="Sign Language Detection", pady=20, font=('Arial', 25, 'bold'), background='#CDCDCD', foreground='#011638')
heading.pack()

top.mainloop()
