import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

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

# Count elements in each cluster for both methods
spectral_counts = {label: sum(spectral_labels == label) for label in set(spectral_labels)}
hierarchical_counts = {label: sum(hierarchical_labels == label) for label in set(hierarchical_labels)}
print("Spectral Clustering counts:", spectral_counts)
print("Hierarchical Clustering counts:", hierarchical_counts)

# Function to label the smaller cluster as "fake"
def label_fake_cluster(labels, counts):
    smaller_cluster = min(counts, key=counts.get)
    return np.array([1 if label == smaller_cluster else 0 for label in labels])

# Label the smaller cluster as "fake" (1 for fake, 0 for normal)
spectral_fake_labels = label_fake_cluster(spectral_labels, spectral_counts)
hierarchical_fake_labels = label_fake_cluster(hierarchical_labels, hierarchical_counts)

# Method to compare how many fakes are common between the two clusters
def compare_fakes(spectral_fake_labels, hierarchical_fake_labels):
    # Element-wise comparison of "fake" labels
    common_fakes = np.sum((spectral_fake_labels == 1) & (hierarchical_fake_labels == 1))
    return common_fakes

common_fakes = compare_fakes(spectral_fake_labels, hierarchical_fake_labels)
print(f"Number of 'fake' images in common between the two clusters: {common_fakes}")

# Method to get the most likely 500 fakes based on distances from cluster centroids
def get_most_likely_fakes(spectral_labels, hierarchical_labels, weighted_features, num_fakes=500):
    # Identify the fake clusters in both methods (smaller cluster in each method)
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
    if len(common_fake_indices) >= num_fakes:
        # If there are already 500 common fakes, return them
        return common_fake_indices[:num_fakes]
    else:
        # If not, combine and sort based on distances
        remaining_needed = num_fakes - len(common_fake_indices)
        
        # Take the top (remaining_needed) from each sorted list
        spectral_top_indices = spectral_sorted_indices[:remaining_needed]
        hierarchical_top_indices = hierarchical_sorted_indices[:remaining_needed]
        
        # Combine common fakes with the top images
        combined_indices = np.concatenate([common_fake_indices, spectral_top_indices, hierarchical_top_indices])
        
        # Return the top 500
        return combined_indices[:num_fakes]

# Get the most likely 500 fake images
most_likely_fakes = get_most_likely_fakes(spectral_labels, hierarchical_labels, weighted_features, num_fakes=500)
print(f"Most likely 500 fake images: {most_likely_fakes}")

# PCA reduction for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(weighted_features)

# 1. Plot Spectral Clustering
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[spectral_labels == 0, 0], reduced_features[spectral_labels == 0, 1], c='blue', alpha=0.6, label="Spectral Cluster 0")
plt.scatter(reduced_features[spectral_labels == 1, 0], reduced_features[spectral_labels == 1, 1], c='red', alpha=0.6, label="Spectral Cluster 1")
plt.title("Spectral Clustering (2D Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# 2. Plot Hierarchical Clustering
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[hierarchical_labels == 0, 0], reduced_features[hierarchical_labels == 0, 1], c='green', alpha=0.6, label="Hierarchical Cluster 0")
plt.scatter(reduced_features[hierarchical_labels == 1, 0], reduced_features[hierarchical_labels == 1, 1], c='purple', alpha=0.6, label="Hierarchical Cluster 1")
plt.title("Hierarchical Clustering (2D Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# 3. Plot 500 most likely fake images and the remaining images
plt.figure(figsize=(10, 8))
remaining_fakes = np.setdiff1d(np.arange(len(reduced_features)), most_likely_fakes)
plt.scatter(reduced_features[remaining_fakes, 0], reduced_features[remaining_fakes, 1], c='blue', alpha=0.6, label="Other Images")
plt.scatter(reduced_features[most_likely_fakes, 0], reduced_features[most_likely_fakes, 1], c='red', alpha=0.6, label="Most Likely Fake Images")
plt.title("2D Visualization of the 500 Most Likely Fake Images and Other Images")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()