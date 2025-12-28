import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import cm

def visualize_gradcam(model, image, class_index, layer_name=None, save_path=None):
    """
    Generates and visualizes Grad-CAM heatmap for a given image and class.
    
    Args:
        model (tf.keras.Model): The trained model.
        image (np.ndarray): The preprocessed input image (shape: (H, W, 3)).
        class_index (int): The target class index.
        layer_name (str, optional): The name of the last convolutional layer. 
                                    If None, it tries to find the last Conv2D layer.
        save_path (str, optional): Path to save the visualization.
    """
    # Create Gradcam object
    # Replace softmax with linear activation for better Grad-CAM results
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    
    # Define score function
    score = CategoricalScore([class_index])
    
    # Generate heatmap
    # image needs to be expanded to batch size 1
    input_image = np.expand_dims(image, axis=0)
    
    # If layer_name is not provided, tf-keras-vis will try to find the last conv layer.
    # However, for stability, passing the layer name is better if known.
    # For now, we let it auto-detect or use -1 if appropriate.
    
    heatmap = gradcam(score, input_image, penultimate_layer=-1)
    
    # Normalize heatmap
    heatmap = heatmap[0]
    
    # Plotting
    # Prepare image for display
    display_image = image.copy()
    if display_image.dtype != np.uint8 and np.max(display_image) > 1.0:
        display_image = display_image / 255.0
        
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(display_image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Grad-CAM Heatmap')
    plt.imshow(heatmap, cmap='jet', alpha=0.8)
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Overlay (Red Part Only)')
    plt.imshow(display_image)
    
    # Create RGBA heatmap for overlay
    # Resize heatmap to match image size first
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize for colormap mapping
    heatmap_norm_full = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8)
    
    # Get colormap
    jet = cm.get_cmap('jet')
    heatmap_rgba = jet(heatmap_norm_full)
    
    # Set transparency: Alpha = 0 where value < 0.5 (filtering blue/green)
    # Also apply global alpha of 0.5 for visible parts
    heatmap_rgba[:, :, 3] = np.where(heatmap_norm_full > 0.5, 0.6, 0.0)
    
    plt.imshow(heatmap_rgba)
    plt.axis('off')

    # 4. Focused Region (Thresholded)
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Thresholding to get the "red part" (high activation)
    # Heatmap is already normalized 0-1 by tf-keras-vis usually, but let's ensure
    heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8)
    mask = heatmap_norm > 0.5
    
    # Apply mask to image
    focused_image = display_image.copy()
    focused_image[~mask] = 0 # Set non-active regions to black
    
    plt.subplot(1, 4, 4)
    plt.title('Focused Region')
    plt.imshow(focused_image)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()