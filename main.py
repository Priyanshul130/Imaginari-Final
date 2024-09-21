import time 
import datetime
import quickdraw



import cv2
import numpy as np


cap = cv2.VideoCapture(0)

lower_marker = np.array([0, 81, 62])
upper_marker = np.array([0, 100, 100])

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
color = colors[0]

canvas = None

block_top, block_bottom = 10, 50
block_width = 100
blocks = [(i * block_width, (i + 1) * block_width) for i in range(6)]

while True:
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

        if block_top <= y <= block_bottom:
            for i, (left, right) in enumerate(blocks):
                if left <= x <= right:
                    color = colors[i]
                    break

        if y > block_bottom:
            cv2.circle(frame, (int(x), int(y)), int(radius), color, -1)
            cv2.circle(canvas, (int(x), int(y)), int(radius), color, -1)
    frame = cv2.add(frame, canvas)

    cv2.imshow("AR Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
