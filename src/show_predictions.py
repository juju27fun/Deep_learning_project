import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import glob

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_image, preprocess_image
from src.interpretability import visualize_gradcam
from src.config import SEED

def main():
    parser = argparse.ArgumentParser(description='Show model predictions for a few images per class.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .h5 model file')
    parser.add_argument('--data_path', type=str, default='train.csv', help='Path to the csv data file')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions_vis', help='Directory to save outputs')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples per class')
    parser.add_argument('--binary', action='store_true', help='Use this flag if the model is binary (2 classes)')
    parser.add_argument('--image_dir', type=str, default='data/train_images', help='Directory containing the images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Update image paths
    # Assuming id_code is just the filename without extension
    df['id_code'] = df['id_code'].apply(lambda x: os.path.join(args.image_dir, f"{x}.png"))
    
    # Handle Binary Case
    if args.binary:
        print("Mapping to Binary Labels (0=No Event, 1=Event)...")
        # 0 -> 0 (No Event)
        # 1, 2, 3, 4 -> 1 (Event)
        df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
        
    df['diagnosis'] = df['diagnosis'].astype(int)
    
    # Get unique classes
    unique_classes = sorted(df['diagnosis'].unique())
    print(f"Found classes: {unique_classes}")
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Iterate per class
    for cls in unique_classes:
        print(f"Processing Class {cls}...")
        class_df = df[df['diagnosis'] == cls]
        
        # Sample images
        if len(class_df) > args.num_samples:
            samples = class_df.sample(args.num_samples, replace=False, random_state=SEED)
        else:
            samples = class_df
            
        for i, (_, row) in enumerate(samples.iterrows()):
            img_path = row['id_code']
            true_label = row['diagnosis']
            
            print(f"  Sample {i+1}: {os.path.basename(img_path)}")
            
            try:
                # Load and Preprocess
                # load_image returns uint8 [0, 255] (H, W, 3)
                original_img = load_image(img_path)
                
                # Preprocess for model -> float32 [0, 255] (H, W, 3)
                input_tensor = preprocess_image(original_img)
                input_img = input_tensor.numpy()
                
                # Predict
                # Expand to batch (1, H, W, 3)
                batch_img = np.expand_dims(input_img, axis=0)
                preds = model.predict(batch_img, verbose=0)
                pred_index = np.argmax(preds[0])
                confidence = preds[0][pred_index]
                
                print(f"    - True: {true_label}, Pred: {pred_index} ({confidence:.2f})")
                
                # Define save path
                filename = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"class{cls}_sample{i+1}_{filename}_true{true_label}_pred{pred_index}.png"
                save_path = os.path.join(args.output_dir, save_name)
                
                # Visualize
                # We pass input_img (float32) to visualize_gradcam. 
                # It handles display normalization if needed.
                visualize_gradcam(
                    model, 
                    input_img, 
                    pred_index, # Visualize the PREDICTED class
                    layer_name=None, 
                    save_path=save_path
                )
                        
            except Exception as e:
                print(f"    - Failed to process {os.path.basename(img_path)}: {e}")
                
    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
