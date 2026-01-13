import numpy as np
import cv2
import os
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import pickle

# CONFIGURATION

# HOG Parameters (standard for face recognition)
HOG_ORIENTATIONS = 9          # Number of gradient orientations
HOG_PIXELS_PER_CELL = (8, 8)  # Cell size
HOG_CELLS_PER_BLOCK = (2, 2)  # Block size
VISUALIZE = False             # Set to True to visualize HOG

# FEATURE EXTRACTION FUNCTIONS

def extract_hog_features(image, visualize=False):
    """
    Extract HOG features from a single image
    
    Args:
        image: Grayscale image (numpy array)
        visualize: Whether to return visualization
        
    Returns:
        features: 1D array of HOG features
        hog_image: (optional) Visualization of HOG
    """
    # Ensure image is in correct format (0-255 range, uint8)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Extract HOG features
    if visualize:
        features, hog_image = hog(
            image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            visualize=True,
            block_norm='L2-Hys'
        )
        # Rescale HOG image for better visualization
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return features, hog_image
    else:
        features = hog(
            image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            visualize=False,
            block_norm='L2-Hys'
        )
        return features


def load_images_and_labels(dataset_path, split='train'):
    """
    Load all images and labels from a dataset split
    
    Args:
        dataset_path: Path to processed dataset
        split: 'train' or 'test'
        
    Returns:
        images: List of images
        labels: List of integer labels
        label_names: List of emotion names
    """
    split_path = os.path.join(dataset_path, split)
    
    # Get emotion folders (sorted for consistency)
    emotion_folders = sorted([f for f in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, f))])
    
    print(f"\nLoading {split} data from {dataset_path}")
    print(f"Emotion classes: {emotion_folders}")
    
    images = []
    labels = []
    
    # Create label mapping
    label_to_int = {emotion: idx for idx, emotion in enumerate(emotion_folders)}
    
    # Load images from each emotion folder
    for emotion in emotion_folders:
        emotion_path = os.path.join(split_path, emotion)
        image_files = sorted([f for f in os.listdir(emotion_path) 
                            if f.endswith('.png')])
        
        print(f"  {emotion}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(emotion_path, img_file)
            
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"    Warning: Could not load {img_path}")
                continue
            
            images.append(img)
            labels.append(label_to_int[emotion])
    
    print(f"Total loaded: {len(images)} images")
    
    return images, labels, emotion_folders


def extract_features_from_dataset(dataset_path, output_file):
    """
    Extract HOG features from entire dataset (train and test)
    
    Args:
        dataset_path: Path to processed dataset
        output_file: Where to save extracted features
    """
    print("\n" + "="*60)
    print(f"EXTRACTING FEATURES FROM: {dataset_path}")
    print("="*60)
    
    # Extract features for training set
    print("\n--- TRAINING SET ---")
    train_images, train_labels, label_names = load_images_and_labels(
        dataset_path, 'train'
    )
    
    print("\nExtracting HOG features from training images...")
    train_features = []
    for i, img in enumerate(train_images):
        if i % 50 == 0:
            print(f"  Processing image {i+1}/{len(train_images)}")
        features = extract_hog_features(img, visualize=False)
        train_features.append(features)
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    
    print(f"\nTraining features shape: {train_features.shape}")
    print(f"  - {train_features.shape[0]} samples")
    print(f"  - {train_features.shape[1]} features per sample")
    
    # Extract features for test set
    print("\n--- TEST SET ---")
    test_images, test_labels, _ = load_images_and_labels(
        dataset_path, 'test'
    )
    
    print("\nExtracting HOG features from test images...")
    test_features = []
    for i, img in enumerate(test_images):
        if i % 50 == 0:
            print(f"  Processing image {i+1}/{len(test_images)}")
        features = extract_hog_features(img, visualize=False)
        test_features.append(features)
    
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    print(f"\nTest features shape: {test_features.shape}")
    print(f"  - {test_features.shape[0]} samples")
    print(f"  - {test_features.shape[1]} features per sample")
    
    # Save features to file
    features_data = {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'label_names': label_names,
        'hog_params': {
            'orientations': HOG_ORIENTATIONS,
            'pixels_per_cell': HOG_PIXELS_PER_CELL,
            'cells_per_block': HOG_CELLS_PER_BLOCK
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(features_data, f)
    
    print(f"\nFeatures saved to: {output_file}")
    
    return features_data


def visualize_hog_features(dataset_path, num_samples=3):
    """
    Create visualization showing original images and their HOG features
    
    Args:
        dataset_path: Path to processed dataset
        num_samples: Number of samples per emotion to visualize
    """
    print("\n" + "="*60)
    print(f"CREATING HOG VISUALIZATIONS")
    print("="*60)
    
    # Load some sample images
    images, labels, label_names = load_images_and_labels(dataset_path, 'train')
    
    # Select samples from each class
    fig, axes = plt.subplots(len(label_names), num_samples * 2, 
                            figsize=(num_samples*4, len(label_names)*2))
    
    fig.suptitle(f'HOG Feature Visualization - {os.path.basename(dataset_path)}',
                fontsize=16, fontweight='bold')
    
    for class_idx in range(len(label_names)):
        # Get images for this class
        class_images = [img for img, lbl in zip(images, labels) if lbl == class_idx]
        
        # Select random samples
        selected_indices = np.random.choice(len(class_images), 
                                          min(num_samples, len(class_images)), 
                                          replace=False)
        
        for sample_idx, img_idx in enumerate(selected_indices):
            img = class_images[img_idx]
            
            # Extract HOG with visualization
            features, hog_image = extract_hog_features(img, visualize=True)
            
            # Plot original image
            col_original = sample_idx * 2
            axes[class_idx, col_original].imshow(img, cmap='gray')
            axes[class_idx, col_original].axis('off')
            if sample_idx == 0:
                axes[class_idx, col_original].set_title(
                    f'{label_names[class_idx]}\nOriginal', 
                    fontsize=10, fontweight='bold'
                )
            else:
                axes[class_idx, col_original].set_title('Original', fontsize=9)
            
            # Plot HOG features
            col_hog = sample_idx * 2 + 1
            axes[class_idx, col_hog].imshow(hog_image, cmap='gray')
            axes[class_idx, col_hog].axis('off')
            axes[class_idx, col_hog].set_title('HOG Features', fontsize=9)
    
    plt.tight_layout()
    
    # Save visualization
    output_name = f'HOG_visualization_{os.path.basename(dataset_path)}.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_name}")


# MAIN EXECUTION

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HOG FEATURE EXTRACTION FOR EMOTION RECOGNITION")
    print("="*60)
    
    # Extract features from CK dataset
    ck_features = extract_features_from_dataset(
        dataset_path='processed_CK_dataset',
        output_file='CK_features.pkl'
    )
    
    # Extract features from JAFFE dataset
    jaffe_features = extract_features_from_dataset(
        dataset_path='processed_JAFFE_dataset',
        output_file='JAFFE_features.pkl'
    )
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    visualize_hog_features('processed_CK_dataset', num_samples=3)
    visualize_hog_features('processed_JAFFE_dataset', num_samples=3)
    
    print("\n" + "="*60)
    print(" FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - CK_features.pkl (HOG features for CK dataset)")
    print("  - JAFFE_features.pkl (HOG features for JAFFE dataset)")
    print("  - HOG_visualization_processed_CK_dataset.png")
    print("  - HOG_visualization_processed_JAFFE_dataset.png")
    print("\nSummary:")
    print(f"  CK Training: {ck_features['train_features'].shape}")
    print(f"  CK Testing: {ck_features['test_features'].shape}")
    print(f"  JAFFE Training: {jaffe_features['train_features'].shape}")
    print(f"  JAFFE Testing: {jaffe_features['test_features'].shape}")
    print("\nReady for model training!")