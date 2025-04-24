import cv2
import numpy as np
import os

wind_data_path = r"C:\Users\admin\Downloads\Asad_Hadi_22BCE1700\wind"
numpies_path = r"C:\Users\admin\Downloads\Asad_Hadi_22BCE1700\numpies"

if not os.path.exists(numpies_path):
    os.makedirs(numpies_path)

image_files = [f for f in os.listdir(wind_data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

template_paths = ['Arrows.jpg', 'Arrows2.jpg','Arrows3.jpg','Arrows4.jpg']
angles = range(0, 360, 5)
white_pixel_threshold = 0.005

def process_image(image_path, filename):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Cropping size
    crop_percentage = 0.6
    crop_width = int(width * crop_percentage)
    crop_height = int(height * crop_percentage)
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    image = image[start_y+100:start_y + crop_height - 350, start_x * 3 + 20: start_x + crop_width - 100]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    arrow_info_array = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))  # 3 channels: i, j, size

    for template_path in template_paths:
        template = cv2.imread(template_path, 0)
        h, w = template.shape[:2]

        for angle in angles:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            rotated_template = cv2.warpAffine(template, M, (w, h))
            result = cv2.matchTemplate(gray_image, rotated_template, cv2.TM_CCOEFF_NORMED)

            threshold = 0.58
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                x1, y1 = pt[0], pt[1]
                x2, y2 = x1 + w, y1 + h

                roi = gray_image[y1:y2, x1:x2]
                white_pixel_count = np.sum(roi == 255)
                total_pixels = roi.size

                white_pixel_ratio = white_pixel_count / total_pixels

                if white_pixel_ratio < white_pixel_threshold:
                    cv2.rectangle(image, pt, (x2, y2), (0, 255, 0), 2)

                    i_component = np.cos(np.deg2rad(angle))  # Horizontal direction
                    j_component = np.sin(np.deg2rad(angle))  # Vertical direction

                    arrow_size = np.sqrt(w ** 2 + h ** 2)

                    arrow_info_array[y1:y2, x1:x2, 0] = i_component  # i-component
                    arrow_info_array[y1:y2, x1:x2, 1] = j_component  # j-component
                    arrow_info_array[y1:y2, x1:x2, 2] = arrow_size   # size of the arrow

        # Resize and save image
        resized_image = cv2.resize(image, (796, 859))
        output_image_path = 'detected_arrows_image_resized.jpg'
        cv2.imwrite(output_image_path, resized_image)
        print(f"Resized image saved as '{output_image_path}'")

        # Resize the arrow information array
        resized_arrow_info_array = np.zeros((859, 796, 3))  # New 796x859 array with 3 channels
        for channel in range(3):
            resized_arrow_info_array[:, :, channel] = cv2.resize(arrow_info_array[:, :, channel], (796, 859))

        # Save the resized arrow info array to the 'numpies' folder with the same filename as the image
        npy_filename = os.path.join(numpies_path, os.path.splitext(filename)[0] + '.npy')
        np.save(npy_filename, resized_arrow_info_array)  # Overwrite if it exists
        print(f"Resized arrow information array saved as '{npy_filename}'")

        # Display the resized image using cv2.imshow()
       

for image_file in image_files:
    image_path = os.path.join(wind_data_path, image_file)
    process_image(image_path, image_file)