import numpy as np
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
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Grad-CAM Heatmap')
    plt.imshow(heatmap, cmap='jet', alpha=0.8)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()