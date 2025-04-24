import cv2
import numpy as np
import matplotlib.pyplot as plt

# Corrected image path
image_path = '/Users/vaibhavagarwal/Desktop/yash/sem5/sih24/extract/yash.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image from path: {image_path}")
    exit()

print("Image loaded successfully.")

# Convert the image to RGB (from BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image (optional - for debugging)
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title('Original Image')
plt.savefig('original_image.png')  # Save the figure to a file
plt.close()  # Close the figure

# Crop the region with the wind arrows and thermal map
# Adjust these coordinates as needed based on the image
wind_region = image_rgb[150:650, 100:1100]  # Example coordinates
colorbar_region = image_rgb[650:700, 100:1100]  # Example coordinates

# Convert wind region to grayscale for arrow detection
wind_gray = cv2.cvtColor(wind_region, cv2.COLOR_RGB2GRAY)

# Apply thresholding to extract arrows (arrows are black, background is colored)

_, thresh = cv2.threshold(wind_gray, 100, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh)
plt.savefig("y.png")
# Detect contours in the thresholded image to identify arrows
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Thresholding and contour detection completed.")

# Draw the detected arrows on the image
arrows_image = wind_region.copy()
cv2.drawContours(arrows_image, contours, -1, (255, 0, 0), 2)

# Display the detected arrows (optional - for debugging)
plt.figure(figsize=(10, 10))
plt.imshow(arrows_image)
plt.title('Detected Arrows (Wind Directions)')
plt.savefig('detected_arrows.png')  # Save the figure to a file
plt.close()  # Close the figure

# Extract color for magnitude from the thermal map
# Create a color-to-speed mapping based on the colorbar
colorbar_rgb = cv2.cvtColor(colorbar_region, cv2.COLOR_BGR2RGB)
# Extract color values along the colorbar for mapping
color_values = colorbar_rgb[25, :]  # Example row for extracting colors
speed_values = np.linspace(0, 20, len(color_values))  # Assuming linear scale from 0 to 20 m/s

# Function to map color to wind speed
def color_to_speed(color, color_values, speed_values):
    distances = np.sqrt(np.sum((color_values - color) ** 2, axis=1))
    return speed_values[np.argmin(distances)]

# Iterate through the wind region and assign wind speed based on the background color
wind_magnitudes = np.zeros(wind_region.shape[:2])

for y in range(wind_region.shape[0]):
    for x in range(wind_region.shape[1]):
        wind_magnitudes[y, x] = color_to_speed(wind_region[y, x], color_values, speed_values)

print("Wind magnitudes extracted.")

# Display the wind magnitudes (thermal map)
plt.figure(figsize=(10, 10))
plt.imshow(wind_magnitudes, cmap='hot', interpolation='nearest')
plt.colorbar(label='Wind Speed (m/s)')
plt.title('Extracted Wind Magnitudes')
plt.savefig('wind_magnitudes.png')  # Save the figure to a file
plt.close()  # Close the figure

# Save the wind direction and magnitude data if needed
np.savetxt('wind_directions.csv', wind_magnitudes, delimiter=',')

print("Wind magnitudes saved to wind_directions.csv.")
