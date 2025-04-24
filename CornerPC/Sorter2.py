import os
import shutil

# Path to the folder containing the new NumPy array files
new_folder_path = r"C:\Users\admin\Downloads\Asad_Hadi_22BCE1700\numpies"
# Path to the output folder where day-wise folders already exist
output_path = r"C:\Users\admin\Downloads\Asad_Hadi_22BCE1700\output"

# Function to check if a file is a NumPy array file (.npy or .npz)
def is_numpy_file(filename):
    return filename.lower().endswith(('.npy', '.npz'))

# Function to move NumPy files from the new folder to the corresponding day-wise folder
def move_numpy_files_to_day_folders(new_folder_path, output_path):
    # List all NumPy array files in the new folder
    new_files = [f for f in os.listdir(new_folder_path) if is_numpy_file(f)]
    
    # Get all day-wise folders
    day_folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]


    if len(new_files) != len(day_folders):
        print("The number of files in the new folder does not match the number of day-wise folders.")
        return

    # Sort both lists to ensure we are moving in a 1-to-1 order (optional, but helps with consistency)
    new_files_sorted = sorted(new_files)
    day_folders_sorted = sorted(day_folders)

    # Move each file to its corresponding day-wise folder
    for new_file, day_folder in zip(new_files_sorted, day_folders_sorted):
        source_file_path = os.path.join(new_folder_path, new_file)
        target_folder_path = os.path.join(output_path, day_folder)

        # Move (or copy) the file to the corresponding day-wise folder
        shutil.move(source_file_path, target_folder_path)
        print(f"Moved {new_file} to {target_folder_path}")

# Call the function to move the NumPy files
move_numpy_files_to_day_folders(new_folder_path, output_path)
