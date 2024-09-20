import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the folder path for the dataset
folder_path = './Q2_dataset'

# Load a grayscale image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

# Compute absolute difference between two images
def compute_difference(image1, image2):
    return np.abs(image1 - image2)

# Display difference images in a grid format
def display_difference_images(diff_images):
    num_images = len(diff_images)
    cols = 3
    rows = (num_images + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    fig.suptitle("Difference Images Grid", fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()

    for i, (label, diff_image) in enumerate(diff_images.items()):
        axes[i].imshow(diff_image, cmap='gray')
        axes[i].set_title(f"Difference: {label}")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Calculate Jaccard Distance between two difference images
def jaccard_distance(diff1, diff2):    
    intersection = np.minimum(diff1, diff2).sum()
    union = np.maximum(diff1, diff2).sum()
    
    if union == 0:
        return 0.0  # Identical images
    return 1 - (intersection / union)

# Main function to execute the program
def main():
    # Paths to images
    image_paths = {
        'OR': f'{folder_path}/OR.jpg',
        'GT': f'{folder_path}/GT.jpg',
        'Algo1': f'{folder_path}/Algo1.jpg',
        'Algo2': f'{folder_path}/Algo2.jpg',
        'Algo3': f'{folder_path}/Algo3.jpg',
        'Algo4': f'{folder_path}/Algo4.jpg',
        'Algo5': f'{folder_path}/Algo5.jpg'
    }

    # Load images
    images = {key: load_image(path) for key, path in image_paths.items()}

    # Compute difference images
    difference_images = {key: compute_difference(images[key], images['OR']) for key in images if key != 'OR'}

    # Calculate Jaccard Distance for each algorithm
    jaccard_distances = {key: jaccard_distance(difference_images['GT'], difference_images[key]) for key in difference_images}
    

    print("Jaccard Distances with respect to GT:")
    for key, value in jaccard_distances.items():
        print(f"{key}: {value}")

    display_difference_images(difference_images)

# Run the program
if __name__ == "__main__":
    main()
