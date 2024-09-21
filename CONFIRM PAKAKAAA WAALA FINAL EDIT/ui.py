import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk

def start_video():
    lower_marker = np.array([80, 100, 100])  # HSV values for teal
    upper_marker = np.array([100, 255, 255])

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]  # Added white color for eraser
    color = colors[0]

    canvas = None
    block_top, block_bottom = 10, 50
    block_width = 100
    blocks = [(i * block_width, (i + 1) * block_width) for i in range(7)]  # Increased range to 7 for eraser block

    prev_center = None
    shape = ""

    cap = cv2.VideoCapture(0)

    def update_frame():
        nonlocal prev_center, shape, canvas

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        marker_mask = cv2.inRange(hsv, lower_marker, upper_marker)
        contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, (left, right) in enumerate(blocks):
            cv2.rectangle(frame, (left, block_top), (right, block_bottom), colors[i], -1)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Simple shape recognition logic
            if area > 1000 and aspect_ratio > 0.5 and aspect_ratio < 2:
                shape = "Rectangle"
            elif area > 1000 and aspect_ratio >= 2:
                shape = "Line"
            elif area > 1000 and aspect_ratio <= 0.5:
                shape = "Circle"
            else:
                shape = ""

            if block_top <= y <= block_bottom:
                for i, (left, right) in enumerate(blocks):
                    if left <= x <= right:
                        color = colors[i]
                        break

            if y > block_bottom:
                if prev_center is not None:
                    for thickness in range(1, 5):
                        cv2.line(canvas, prev_center, center, (color[0] + 20, color[1] + 20, color[2] + 20), thickness)
                        cv2.line(canvas, prev_center, center, color, thickness=1)
                prev_center = center

        else:
            prev_center = None

        # Display the recognized shape
        cv2.putText(frame, shape, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        frame = cv2.add(frame, canvas)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        root.after(10, update_frame)

    def save_painting():
        filename = 'painting.png'
        if canvas is not None:
            cv2.imwrite(filename, canvas)
            messagebox.showinfo("Success", f"Painting saved as {filename}")
        else:
            messagebox.showerror("Error", "No painting to save")

    root = tk.Tk()
    root.title("IMAGINARI - Video Capture and Prediction")
    
    video_label = Label(root)
    video_label.pack()

    save_button = tk.Button(root, text="Save Painting", command=save_painting)
    save_button.pack()

    root.after(10, update_frame)
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

start_video()
