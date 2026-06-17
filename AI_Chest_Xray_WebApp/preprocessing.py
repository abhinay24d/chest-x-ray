
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
    Extracts CLAHE + HOG + LBP features from a chest X-ray image with
    automatic border cropping and Gaussian noise reduction for web image robustness.
    
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
        
    # Convert color channel to Grayscale
    img = img.convert("L")
    img_raw_array = np.array(img)
    
    # 2. Auto-crop black borders
    # Find pixels where intensity is above a threshold (e.g. 10/255)
    row_sums = np.mean(img_raw_array, axis=1)
    col_sums = np.mean(img_raw_array, axis=0)
    rows = np.where(row_sums > 10)[0]
    cols = np.where(col_sums > 10)[0]
    
    if len(rows) > 0 and len(cols) > 0:
        # Crop the PIL image to exclude solid black boundaries
        img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
        
    # 3. Resize to standardized 128x128 grid
    img = img.resize((128, 128))
    img_array = np.array(img)
    
    # 4. Standardize Contrast using CLAHE
    # equalize_adapthist expects a float array in [0, 1] range
    img_float = img_array / 255.0
    img_equalized = equalize_adapthist(img_float, clip_limit=0.03)
    
    # 5. Apply Gaussian Denoising to smooth out JPEG compression artifacts
    from skimage.filters import gaussian
    img_denoised = gaussian(img_equalized, sigma=1.0)
    
    # Convert back to uint8 for LBP and HOG
    img_equalized_uint8 = (img_denoised * 255.0).astype(np.uint8)
    
    # 6. Extract shape/edge features using HOG
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
    
    # 7. Extract texture features using LBP (Local Binary Patterns)
    # P=8 (points), R=1 (radius)
    lbp = local_binary_pattern(img_equalized_uint8, P=8, R=1, method='uniform')
    
    # Compute normalized histogram of uniform LBP patterns (which has 10 bins)
    # Using np.histogram with bins from 0 to 10. Density=True returns probabilities.
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    
    # 8. Concatenate features into a unified feature vector (1568 + 10 = 1578 dimensions)
    features = np.concatenate([hog_features, lbp_hist])
    
    return features
