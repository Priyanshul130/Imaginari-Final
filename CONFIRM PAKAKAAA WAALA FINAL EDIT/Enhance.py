import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_marker = np.array([80, 100,100])
upper_marker = np.array([100, 255, 255])

# Colors: added white as eraser
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]  # White color for eraser
color = colors[0]

canvas = None
history = []  # History to store canvas for undo functionality

block_top, block_bottom = 10, 50
block_width = 100
blocks = [(i * block_width, (i + 1) * block_width) for i in range(7)]  # Added eraser block

prev_center = None  # Variable to store the previous position of the marker
shape = ""  # Variable to store the recognized shape
draw = True  # Flag to track if drawing should be done

def detect_shape(cnt):
    """Detect the shape of a contour."""
    shape = ""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif len(approx) > 5:
        shape = "Circle"
    
    return shape

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

        # Shape recognition
        shape = detect_shape(cnt)

        if block_top <= y <= block_bottom:
            for i, (left, right) in enumerate(blocks):
                if left <= x <= right:
                    color = colors[i]
                    draw = False if i == 6 else True  # If white (eraser) selected, set draw to False
                    break

        if y > block_bottom and draw:
            if prev_center is not None:
                # Save the canvas state for undo
                history.append(canvas.copy())
                
                if color == (255, 255, 255):  # Eraser functionality
                    cv2.line(canvas, prev_center, center, (0, 0, 0), thickness=10)  # Eraser thickness is larger
                else:
                    # Smoother drawing lines
                    for thickness in range(1, 5):
                        cv2.line(canvas, prev_center, center, color, thickness)

            prev_center = center

    else:
        prev_center = None

    # Display the recognized shape
    if shape:
        cv2.putText(frame, shape, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    frame = cv2.add(frame, canvas)

    cv2.imshow("IMAGINARI", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('s'):  # Press 's' to save the painting
        filename = 'painting.png'
        cv2.imwrite(filename, canvas)
        print(f"Painting saved as {filename}")
        cv2.putText(frame, "Saved!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("IMAGINARI", frame)
        cv2.waitKey(500) 

    if key & 0xFF == ord('u') and len(history) > 0: 
        canvas = history.pop()
        print("Undo the last stroke")

cap.release()
cv2.destroyAllWindows()
