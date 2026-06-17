
"""
==========================================================================
Chest X-Ray Preprocessing and Classical Feature Extraction Pipeline
==========================================================================
This module implements the exact same CLAHE (contrast equalization), HOG,
and LBP feature extraction pipeline for both training and real-time inference.
"""

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.exposure import equalize_adapthist

def extract_features(img_or_path):
    """
    Extracts CLAHE + HOG + LBP features from a chest X-ray image.
    

    
    Parameters:
        img_or_path (str or PIL.Image.Image): Filepath or loaded PIL Image object.
        
    Returns:
        np.ndarray: A 1D feature vector of shape (1578,) containing HOG and LBP descriptors.
    """
    # 1. Load image if path is given
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path
        
    # 2. Standardize color channel to Grayscale & resize to 128x128
    img = img.convert("L")
    img = img.resize((128, 128))
    img_array = np.array(img)
    
    # 3. Standardize Contrast using CLAHE
    # equalize_adapthist expects a float array in [0, 1] range
    img_float = img_array / 255.0
    img_equalized = equalize_adapthist(img_float, clip_limit=0.03)
    
    # Convert back to uint8 for robust feature calculation (required by LBP)
    img_equalized_uint8 = (img_equalized * 255.0).astype(np.uint8)
    
    # 4. Extract shape/edge features using HOG
    # Using 8 orientations, 16x16 pixels per cell, 2x2 cells per block
    # For a 128x128 image, this yields:
    #   (128/16 - 1) * (128/16 - 1) = 7 * 7 = 49 blocks
    #   49 blocks * (2 * 2 cells/block) * 8 orientations = 1568 features.
    hog_features = hog(
        img_equalized_uint8,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    # 5. Extract texture features using LBP (Local Binary Patterns)
    # P=8 (points), R=1 (radius)
    lbp = local_binary_pattern(img_equalized_uint8, P=8, R=1, method='uniform')
    
    # Compute normalized histogram of uniform LBP patterns (which has 10 bins)
    # Using np.histogram with bins from 0 to 10. Density=True returns probabilities.
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    
    # 6. Concatenate features into a unified feature vector (1568 + 10 = 1578 dimensions)
    features = np.concatenate([hog_features, lbp_hist])
    
    return features
