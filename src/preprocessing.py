import cv2
import tensorflow as tf
import numpy as np
from src.config import IMG_SIZE

def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from the image (Ben Graham's method).
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # Image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def load_image(path):
    """
    Loads an image from a given path, applies circular cropping and resizing.
    
    Args:
        path (str): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image in RGB format.
    """
    # Handle numpy array input (from tf.numpy_function)
    if isinstance(path, np.ndarray):
        path = path.item()
    if isinstance(path, bytes):
        path = path.decode('utf-8')

    # Read image using OpenCV
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply Ben Graham's preprocessing (Crop black borders)
    image = crop_image_from_gray(image)
    
    # Resize to target size using OpenCV (better quality/control than tf.image.resize for this step)
    image = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Ben Graham's Color Normalization (Optional but often helpful)
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
    
    return image

def preprocess_image(image, label=None):
    """
    Prepares the image for the model.
    Note: EfficientNet expects [0, 255] inputs, so we DO NOT divide by 255.0 here.
    
    Args:
        image (np.ndarray or tf.Tensor): The input image.
        label (int, optional): The class label. Defaults to None.
        
    Returns:
        tf.Tensor: The preprocessed image.
        int (optional): The label, if provided.
    """
    # Ensure image is float32 for TensorFlow operations, but keep scale [0, 255]
    image = tf.cast(image, tf.float32)
    
    # Explicit resize again to ensure tensor shape is set (needed for batching)
    image = tf.image.resize(image, IMG_SIZE)
    
    if label is not None:
        return image, label
    return image

def augment_image(image, label=None):
    """
    Applies random augmentations to the image.
    
    Args:
        image (tf.Tensor): The input image.
        label (int, optional): The class label. Defaults to None.
        
    Returns:
        tf.Tensor: The augmented image.
        int (optional): The label, if provided.
    """
    # Random flip left-right
    image = tf.image.random_flip_left_right(image)
    
    # Random flip up-down
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random rotation (90 degrees)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    
    # Random Zoom/Crop (Simulated by resize with crop)
    # This is a bit expensive in TF graph, skipping for speed unless necessary
    
    if label is not None:
        return image, label
    return image