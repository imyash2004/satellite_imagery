import os
from datetime import datetime
import shutil

# Paths to BT1 folders and output folder
bt_data_path = r"C:\Users\admin\Documents\BT1"
output_path = r"C:\Users\admin\Documents\output"  # Folder to store date-based folders

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Function to extract the day of the year from BT image filename
def get_bt_day(bt_filename):
    try:
        # Example BT filename: "3RIMG_30JAN2024_2215_L1C_ASIA_MER_BT_IR1_TEMP_V01R00.jpg"
        parts = bt_filename.split('_')
        if len(parts) > 1:
            day_str = parts[1][0:2]  # Example: "30"
            month_str = parts[1][2:5]  # Example: "JAN"
            year_str = parts[1][5:9]  # Example: "2024"
            
            # Convert to a datetime object to get the day of the year
            bt_datetime = datetime.strptime(f'{day_str}{month_str}{year_str}', '%d%b%Y')
            return bt_datetime.timetuple().tm_yday
        else:
            print(f"Invalid BT filename format: {bt_filename}")
            return None  # Handle error case
    except (IndexError, ValueError) as e:
        print(f"Error parsing BT filename: {bt_filename} ({e})")
        return None  # Handle error case

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

# Function to organize BT images by date into separate folders
def organize_bt_by_date(bt_data_path, output_path):
    # Iterate over each month folder in BT1
    for month_folder in os.listdir(bt_data_path):
        month_path = os.path.join(bt_data_path, month_folder)
        if os.path.isdir(month_path):
            # List all BT images in the month folder, filter only image files, and sort by day
            bt_images = [f for f in os.listdir(month_path) if is_image_file(f)]
            valid_bt_images = [f for f in bt_images if get_bt_day(f) is not None]
            bt_images_sorted = sorted(valid_bt_images, key=lambda f: get_bt_day(f))

            # For each BT image, create a folder based on the day and move the image there
            for bt_file in bt_images_sorted:
                bt_day_num = get_bt_day(bt_file)

                if bt_day_num is not None:
                    # Create a folder for the specific day (e.g., "Day_30")
                    day_folder = os.path.join(output_path, f"Day_{bt_day_num}")
                    os.makedirs(day_folder, exist_ok=True)

                    # Copy the BT image to the corresponding day folder
                    bt_file_path = os.path.join(month_path, bt_file)
                    shutil.copy(bt_file_path, day_folder)

# Call the function to organize BT images into date-based folders
organize_bt_by_date(bt_data_path, output_path)

print(f"BT images have been organized into date-based folders at: {output_path}")
