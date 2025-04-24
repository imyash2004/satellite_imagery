import numpy as np
import cv2
import pandas as pd

# Load the images
wind_image = cv2.imread('/Users/vaibhavagarwal/Desktop/yash/sem5/sih24/mapping/i1.jpg')
cloud_image = cv2.imread('/Users/vaibhavagarwal/Desktop/yash/sem5/sih24/mapping/i2.jpg')  # Note: i2.jpg is the cloud image

# Resize wind_image to match cloud_image dimensions
wind_image_resized = cv2.resize(wind_image, (cloud_image.shape[1], cloud_image.shape[0]))

# Placeholder function to extract wind speed (needs to be implemented)
def extract_wind_speed(wind_image):
    # Convert color map to wind speed (implement color to speed mapping)
    speed_map = np.zeros(wind_image.shape[:2])
    # Populate with actual wind speed values
    return speed_map

# Placeholder function to extract wind direction (needs to be implemented)
def extract_wind_direction(wind_image):
    # Calculate wind direction in radians (implement arrow detection)
    direction_map = np.zeros(wind_image.shape[:2])
    return direction_map

# Placeholder function to extract cloud density (needs to be implemented)
def extract_cloud_density(cloud_image):
    # Convert to grayscale or binary to represent cloud presence/density
    cloud_density_map = cv2.cvtColor(cloud_image, cv2.COLOR_BGR2GRAY)
    return cloud_density_map

# Extract wind speed, direction, and cloud density
wind_speed = extract_wind_speed(wind_image_resized)
wind_direction = extract_wind_direction(wind_image_resized)
cloud_density = extract_cloud_density(cloud_image)

# Combine the features into a matrix
combined_features = np.stack((wind_speed, wind_direction, cloud_density), axis=-1)

# Flatten the feature matrix to prepare for CSV storage
flattened_features = combined_features.reshape(-1, 3)  # Shape: (height*width, 3)

# Convert the flattened features into a DataFrame for CSV storage
df = pd.DataFrame(flattened_features, columns=['Wind_Speed', 'Wind_Direction', 'Cloud_Density'])

# Save the DataFrame to a CSV file
output_csv_path = 'wind_cloud_features.csv'
df.to_csv(output_csv_path, index=False)

print(f"Feature matrix saved to {output_csv_path}")
