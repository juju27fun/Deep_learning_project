import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_image, preprocess_image
from src.interpretability import visualize_gradcam

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for trained models.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .h5 model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to an image or directory of images')
    parser.add_argument('--output_dir', type=str, default='gradcam_outputs', help='Directory to save outputs')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process if image_path is a directory')
    parser.add_argument('--binary', action='store_true', help='Use this flag if the model is binary (2 classes)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Gather images
    image_files = []
    if os.path.isdir(args.image_path):
        # Support common image formats
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(args.image_path, ext)))
        
        if not image_files:
            print(f"No images found in {args.image_path}")
            return
            
        # Sample images
        if len(image_files) > args.num_samples:
            image_files = np.random.choice(image_files, args.num_samples, replace=False)
    else:
        if os.path.exists(args.image_path):
            image_files = [args.image_path]
        else:
            print(f"Image path not found: {args.image_path}")
            return

    print(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        print(f"Generating Grad-CAM for {os.path.basename(img_path)}...")
        
        try:
            # Load and Preprocess
            # load_image returns a numpy array in RGB [0, 255]
            original_img = load_image(img_path)
            
            # Preprocess for model (Resize, Float32, etc.)
            # Note: preprocess_image returns tensor, so we convert to numpy
            input_tensor = preprocess_image(original_img)
            input_img = input_tensor.numpy()
            
            # Predict
            # Expand dims for batch
            batch_img = np.expand_dims(input_img, axis=0)
            preds = model.predict(batch_img, verbose=0)
            pred_index = np.argmax(preds[0])
            confidence = preds[0][pred_index]
            
            print(f"  Prediction: Class {pred_index} ({confidence:.2f})")
            
            # Save path
            filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.output_dir, f"gradcam_{filename}_class{pred_index}.png")
            
            # Generate Visualization
            # Note: visualize_gradcam expects the original image for overlay (RGB [0, 255])
            # But technically it uses the input_img for gradient calculation?
            # Actually visualize_gradcam implementation:
            #   input_image = np.expand_dims(image, axis=0) -> Passes 'image' to gradcam
            #   It also uses 'image' for plt.imshow().
            #   
            #   If we pass the preprocessed image (which is resized), the overlay works.
            #   Original image might be different size.
            #   Let's pass the preprocessed image (converted back to uint8 for visualization if needed, 
            #   but plt.imshow handles floats [0,1] or ints [0,255]).
            #   
            #   preprocess_image in this codebase does NOT normalize to [0,1] (it keeps [0,255] float).
            #   So we should cast to int for display or normalize.
            #   
            #   Checking interpretability.py:
            #   plt.imshow(image) -> If floats > 1, matplotlib clips or errors? No, if it's float it expects [0,1].
            #   If it's int it expects [0,255].
            #   preprocess_image returns float32 [0, 255].
            #   So we should cast to uint8 for visualization validity.
            
            # Note: visualize_gradcam expects the original image for overlay (RGB [0, 255])
            # But technically it uses the input_img for gradient calculation?
            # Actually visualize_gradcam implementation:
            #   input_image = np.expand_dims(image, axis=0) -> Passes 'image' to gradcam
            #   It also uses 'image' for plt.imshow().
            #   
            #   We now pass the float32 array (0-255) and let visualize_gradcam handle display normalization.
            
            visualize_gradcam(
                model, 
                input_img, # Pass float32 array
                pred_index, 
                layer_name=None, 
                save_path=save_path
            )
            
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
