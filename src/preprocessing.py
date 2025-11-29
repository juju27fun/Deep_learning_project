import cv2
import tensorflow as tf
import numpy as np
from src.config import IMG_SIZE, BATCH_SIZE

def load_image(path):
    """
    Loads an image from a given path.
    
    Args:
        path (str): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image in RGB format.
    """
    # Read image using OpenCV
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image, label=None):
    """
    Resizes the image to IMG_SIZE and normalizes pixel values to [0, 1].
    
    Args:
        image (np.ndarray or tf.Tensor): The input image.
        label (int, optional): The class label. Defaults to None.
        
    Returns:
        tf.Tensor: The preprocessed image.
        int (optional): The label, if provided.
    """
    # Resize image
    image = tf.image.resize(image, IMG_SIZE)
    
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
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
    
    # Random rotation (90 degrees)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    
    if label is not None:
        return image, label
    return image

def create_dataset(dataframe, is_training=True):
    """
    Creates a tf.data.Dataset from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing 'id_code' and 'diagnosis' columns.
        is_training (bool, optional): Whether to apply augmentation and shuffling. Defaults to True.
        
    Returns:
        tf.data.Dataset: The created dataset.
    """
    paths = dataframe['id_code'].values
    labels = dataframe['diagnosis'].astype(int).values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img, label = preprocess_image(img, label)
        return img, label
    
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(dataframe))
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
