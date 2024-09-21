import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_marker = np.array([50, 50, 50])
upper_marker = np.array([150, 150, 150])

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]  # Added white color for eraser
color = colors[0]

canvas = None
#creating block
block_top, block_bottom = 10, 50
block_width = 100
blocks = [(i * block_width, (i + 1) * block_width) for i in range(7)]  # Increased range to 7 for eraser block

prev_center = None  # Variable to store the previous position of the marker
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
        center = (int(x), int(y))

        if block_top <= y <= block_bottom:
            for i, (left, right) in enumerate(blocks):
                if left <= x <= right:
                    color = colors[i]
                    break

        if y > block_bottom:
            if prev_center is not None:
                # Highlighting effect: draw multiple lines with different shades
                for thickness in range(1, 5):  # Increase the range to make it more pronounced
                    cv2.line(canvas, prev_center, center, (color[0] + 20, color[1] + 20, color[2] + 20), thickness)
                    cv2.line(canvas, prev_center, center, color, thickness=1)
            prev_center = center

    else:
        prev_center = None

    frame = cv2.add(frame, canvas)

    cv2.imshow("IMAGINARI", frame)
    key = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('s'):  # Press 's' to save the painting

        filename = 'painting.png'
        cv2.imwrite(filename, canvas)
        print(f"Painting saved as {filename}")

cap.release()
cv2.destroyAllWindows()