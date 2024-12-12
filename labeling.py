import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from PIL import Image

# Parameters
image_folder = "allfrog"
num_clusters = 2  # Normal and anomalous
image_size = (224, 224)  # ResNet input size

# Load ResNet for feature extraction
def get_feature_extractor():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the classification head
    model.eval()
    return model

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze(0).numpy()  # Remove batch dimension

# Load all images and extract features
print("Extracting features from images...")
model = get_feature_extractor()
image_features = []
image_paths = []

for root, _, files in os.walk(image_folder):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(root, file)
            features = extract_features(image_path, model)
            image_features.append(features)
            image_paths.append(image_path)

image_features = np.array(image_features)

# Weight computation (importance of each patch)
def compute_weights(features):
    weights = np.linalg.norm(features, axis=1)  # Use L2 norm as a simple proxy
    return weights / np.sum(weights)

weights = compute_weights(image_features)
weighted_features = image_features * weights[:, np.newaxis]  # Weighted embeddings

# Spectral Clustering
print("Applying Spectral Clustering...")
spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral_clustering.fit_predict(weighted_features)

# Hierarchical Clustering
print("Applying Hierarchical Clustering...")
hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters)
hierarchical_labels = hierarchical_clustering.fit_predict(weighted_features)

def get_most_likely_fakes(spectral_labels, hierarchical_labels, weighted_features, num_fakes=500):
    # Determine which clusters are likely to be fake
    spectral_fake_label = np.argmin([np.sum(spectral_labels == 0), np.sum(spectral_labels == 1)])  # Smaller cluster in Spectral
    hierarchical_fake_label = np.argmin([np.sum(hierarchical_labels == 0), np.sum(hierarchical_labels == 1)])  # Smaller cluster in Hierarchical
    
    # Get the indices of the fake images from both methods
    spectral_fake_indices = np.where(spectral_labels == spectral_fake_label)[0]
    hierarchical_fake_indices = np.where(hierarchical_labels == hierarchical_fake_label)[0]
    
    # Calculate centroids of the fake clusters for both methods
    spectral_centroid = np.mean(weighted_features[spectral_fake_indices], axis=0)
    hierarchical_centroid = np.mean(weighted_features[hierarchical_fake_indices], axis=0)
    
    # Calculate distances of all images to both centroids
    spectral_distances = np.linalg.norm(weighted_features - spectral_centroid, axis=1)
    hierarchical_distances = np.linalg.norm(weighted_features - hierarchical_centroid, axis=1)
    
    # Get the indices of the fakes from Spectral and Hierarchical clustering
    spectral_sorted_indices = np.argsort(spectral_distances)
    hierarchical_sorted_indices = np.argsort(hierarchical_distances)
    
    # Select the top 500 fakes, prioritize common fakes, but also include a mix from both methods
    common_fake_indices = np.intersect1d(spectral_fake_indices, hierarchical_fake_indices)
    
    # If there are already 500 common fakes, return them
    if len(common_fake_indices) >= num_fakes:
        return common_fake_indices[:num_fakes]
    else:
        # If not, combine and sort based on distances
        remaining_needed = num_fakes - len(common_fake_indices)
        
        # Take the top (remaining_needed) from each sorted list
        spectral_top_indices = spectral_sorted_indices[:remaining_needed]
        hierarchical_top_indices = hierarchical_sorted_indices[:remaining_needed]
        
        # Combine common fakes with the top images
        combined_indices_set = set(common_fake_indices)  # Using a set to avoid duplicates
        
        # Add the top indices from both methods to the set
        combined_indices_set.update(spectral_top_indices)
        combined_indices_set.update(hierarchical_top_indices)
        
        # Convert the set back to a list and sort it
        combined_indices = list(combined_indices_set)
        
        # Return the top 500 unique indices
        return combined_indices[:num_fakes]

# Get the most likely 500 fake images
most_likely_fakes = get_most_likely_fakes(spectral_labels, hierarchical_labels, weighted_features, num_fakes=500)
# Sort image paths by their filenames (lexicographically)
sorted_image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# Create a dictionary to store image names and labels
image_labels = {}

# Label each image as "good" or "bad" based on its index
bad_count = 0
good_count = 0
for i, image_path in enumerate(sorted_image_paths):
    label="good"
    for j in range(len(most_likely_fakes)):
        if i == int(most_likely_fakes[j]):
            label = "bad" 
    
    # Store image name and label
    image_labels[os.path.basename(image_path)] = label
    
    # Count the number of bad and good labels
    if label == "bad":
        bad_count += 1
    else:
        good_count += 1

# Print the number of bad and good images
print(f"Number of 'bad' images: {bad_count}")
print(f"Number of 'good' images: {good_count}")

# Write labels to the labels.txt file (in the order of the sorted filenames)
with open("labels.txt", "w") as file:
    for image_name, label in image_labels.items():
        file.write(f"{label}\n")

print("Labels file 'labels.txt' has been generated.")
