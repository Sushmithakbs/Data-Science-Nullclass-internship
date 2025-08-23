import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_drowsiness(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    sleep_count = 0
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            sleep_count += 1
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    messagebox.showinfo("Result", f"Sleeping people: {sleep_count}")

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_drowsiness(file_path)

root = tk.Tk()
root.title("Drowsiness Detection")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=20)

root.mainloop()