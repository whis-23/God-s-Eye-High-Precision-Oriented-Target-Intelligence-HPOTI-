import numpy as np
import cv2

def log_transform(image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compress the dynamic range of SAR imagery using log-transformation.
    formula: I_out = 10 * log10(I_in + epsilon)
    
    Args:
        image: Input SAR image (numpy array).
        epsilon: Small constant to avoid log(0).
        
    Returns:
        Log-transformed image, normalized to [0, 255] for processing.
    """
    # Ensure image is float
    image_float = image.astype(np.float32)
    
    # Apply log transformation
    log_img = 10.0 * np.log10(image_float + epsilon)
    
    # Normalize to 0-255 range for consistency with standard vision models
    min_val = np.min(log_img)
    max_val = np.max(log_img)
    
    if max_val > min_val:
        normalized_img = (log_img - min_val) / (max_val - min_val) * 255.0
    else:
        normalized_img = log_img
        
    return normalized_img.astype(np.uint8)

def lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Lee filter to reduce speckle noise while preserving edges.
    
    Args:
        image: Input grayscale SAR image (numpy array).
        window_size: Size of the local window (must be odd).
        
    Returns:
        Filtered image.
    """
    img_float = image.astype(np.float32)
    
    # Calculate local mean
    local_mean = cv2.blur(img_float, (window_size, window_size))
    
    # Calculate local mean of squared image
    local_mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    
    # Calculate local variance
    local_var = local_mean_sq - local_mean**2
    
    # Estimate noise variance (global noise floor)
    overall_var = np.var(img_float)
    
    # Calculate weighting coefficient K
    # K = local_var / (local_var + noise_var)
    # Avoiding division by zero
    k = local_var / (local_var + overall_var + 1e-6)
    
    # Filtered value
    result = local_mean + k * (img_float - local_mean)
    
    return np.clip(result, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Simple test
    test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    log_img = log_transform(test_img)
    filtered_img = lee_filter(test_img)
    print("Preprocessing filters smoke test passed.")
