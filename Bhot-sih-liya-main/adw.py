import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'i1.jpg'  # Change to the path of your image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to detect all edges
edges = cv2.Canny(gray, 50, 150)

# Find contours from the detected edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through each contour to detect arrows based on their shape
for contour in contours:
    # Fit a bounding box around the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Calculate the width and height of the bounding box
    width, height = rect[1]

    # Filter based on the aspect ratio of the bounding box (arrows are long and narrow)
    aspect_ratio = max(width, height) / min(width, height)

    # Set thresholds for detecting arrows based on shape
    if 3 < aspect_ratio < 10 and min(width, height) > 5:  # Adjust for arrow-like shapes
        # Draw the rectangle on the original image
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        # Calculate the i and j components (direction and length of the arrow)
        angle = rect[2]
        arrow_length = max(width, height)  # Length of the arrow (major axis)
        i_component = arrow_length * np.cos(np.deg2rad(angle))
        j_component = arrow_length * np.sin(np.deg2rad(angle))

        # Display the arrow's i and j components
        center = (int(rect[0][0]), int(rect[0][1]))
        print(f"Arrow at {center} has i component: {i_component:.2f}, j component: {j_component:.2f}")

# Display the result
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Arrows Based on Shape and Aspect Ratio')
plt.axis('off')
plt.show()
