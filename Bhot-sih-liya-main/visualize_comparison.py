import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Load sample images
def load_image(path):
    return np.array(Image.open(path).convert('L')) / 255.0

# Create visualization
def visualize_comparison(input1_path, input2_path, generated_path, ground_truth_path=None):
    input1 = load_image(input1_path)
    input2 = load_image(input2_path)
    generated = load_image(generated_path)
    
    plt.figure(figsize=(20, 5))
    
    # Input image 1
    plt.subplot(1, 4, 1)
    plt.imshow(input1, cmap='gray')
    plt.title("Input Image 1")
    plt.axis('off')
    
    # Generated image
    plt.subplot(1, 4, 2)
    plt.imshow(generated, cmap='gray')
    plt.title("Generated Image")
    plt.axis('off')
    
    # Ground truth (if available)
    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = load_image(ground_truth_path)
        plt.subplot(1, 4, 3)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
    
    # Input image 2
    plt.subplot(1, 4, 4)
    plt.imshow(input2, cmap='gray')
    plt.title("Input Image 2")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage (modify paths as needed)
if __name__ == "__main__":
    # Replace these paths with actual image paths
    input1 = "MaskedImages/masked_image1.png"
    input2 = "MaskedImages/masked_image2.png"
    generated = "GeneratedImages/generated_image_1.png"
    ground_truth = "Dataset/middle_frame.png"  # If available
    
    visualize_comparison(input1, input2, generated, ground_truth)
