import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
import cv2
import random

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define class labels (truck -> 9, frog -> 6)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create directories to save poisoned images
poisoned_dir = 'poisoned_images'
os.makedirs(poisoned_dir, exist_ok=True)
os.makedirs(os.path.join(poisoned_dir, 'frogs/fake'), exist_ok=True)
os.makedirs(os.path.join(poisoned_dir, 'frogs/real'), exist_ok=True)

# Define all poisoning techniques
techniques = [
    'gradient-matching', 
    #'watermark', 
    'poison-frogs', 'metapoison',
    #'hidden-trigger', 'convex-polytope', 
    'bullseye', 
    'patch'
]

# Function to generate a poisoned image
def apply_poison(image, technique='gradient-matching'):
    if technique == 'gradient-matching':
        return apply_gradient_matching(image)
    elif technique == 'watermark':
        return apply_watermark(image)
    elif technique == 'poison-frogs':
        return apply_poison_frogs(image)
    elif technique == 'metapoison':
        return apply_metapoison(image)
    elif technique == 'hidden-trigger':
        return apply_hidden_trigger(image)
    elif technique == 'convex-polytope':
        return apply_convex_polytope(image)
    elif technique == 'bullseye':
        return apply_bullseye(image)
    elif technique == 'patch':
        return apply_patch(image)
    else:
        return image

# Example of gradient-matching technique
def apply_gradient_matching(image):
    noisy_image = image + np.random.uniform(-0.1, 0.1, size=image.shape)
    return np.clip(noisy_image, 0, 255)

# Example of watermark technique
def apply_watermark(image):
    watermark = np.zeros_like(image)
    cv2.putText(watermark, "TRUCK", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    poisoned_image = cv2.addWeighted(image, 1, watermark, 0.1, 0)
    return poisoned_image

# Example of poison-frogs technique
def apply_poison_frogs(image):
    frog_image = image.copy()
    frog_image[..., 1] = frog_image[..., 1] * 1.2  # Boost the green channel
    return np.clip(frog_image, 0, 255)

# Example of metapoison technique
def apply_metapoison(image):
    image[:5, :5] = 0  # Clear a small region (trigger)
    return image

# Example of hidden-trigger technique
def apply_hidden_trigger(image):
    image[-10:, -10:] = 255  # Trigger in the bottom-right corner
    return image

# Example of convex-polytope technique
def apply_convex_polytope(image):
    image[15:25, 15:25] = 255  # Square patch as a convex polytope
    return image

def apply_bullseye(image):
    # Ensure the image is in the correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Normalize if needed

    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Define the center of the bullseye
    center = (image.shape[1] // 2, image.shape[0] // 2)  # (x, y)

    # Draw concentric circles (bullseye)
    for i in range(5, 50, 10):
        cv2.circle(image, center, i, (0, 255, 0), 1)

    return image

# Example of patch technique
def apply_patch(image):
    image[10:30, 10:30] = 255  # White patch in the top-left corner
    return image

# Function to save poisoned and real frog images
def save_poisoned_and_real_images(x, y, techniques, save_dir):
    # Select frog images (label 6)
    frog_images = np.where(y == 6)[0]  # Only process "frog" images (label 6)
    random.shuffle(frog_images)  # Shuffle the frog images to randomize

    # Half of the frog images will be poisoned, half will remain real
    poisoned_frogs = frog_images[:len(frog_images) // 2]  # First half for poisoning
    real_frogs = frog_images[len(frog_images) // 2:]  # Second half as real

    num_techniques = len(techniques)
    num_poisoned_images = len(poisoned_frogs)  # Total number of poisoned frog images
    num_images_per_technique = num_poisoned_images // num_techniques  # Number of images per technique

    # Apply each technique to the appropriate number of poisoned frogs
    for technique in techniques:
        technique_start_idx = techniques.index(technique) * num_images_per_technique
        technique_end_idx = technique_start_idx + num_images_per_technique

        for i in range(technique_start_idx, technique_end_idx):
            idx = poisoned_frogs[i]
            poisoned_image = apply_poison(x[idx], technique=technique)
            poisoned_image_pil = image.array_to_img(poisoned_image)
            
            # Save the poisoned image with technique in the filename
            file_path = os.path.join(save_dir, 'frogs/fake', f"poisoned_frog_{technique}_{idx}_fake.png")
            poisoned_image_pil.save(file_path)
            
            if (i - technique_start_idx + 1) % 10 == 0:
                print(f"Saved poisoned frog image {technique} {i - technique_start_idx + 1} of {num_images_per_technique}")


    # Save real frog images
    for i, idx in enumerate(real_frogs):
        real_frog_image_pil = image.array_to_img(x[idx])
        file_path = os.path.join(save_dir, 'frogs/real', f"real_frog_{idx}.png")
        real_frog_image_pil.save(file_path)
        if i % 10 == 0:
            print(f"Saved real frog image {i + 1}")

# Apply poisoning techniques and save poisoned and real frog images
save_poisoned_and_real_images(x_train, y_train, techniques, poisoned_dir)

print("Poisoned and real frog images saved.")
