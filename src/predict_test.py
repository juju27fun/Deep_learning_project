import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_image, preprocess_image, augment_image

def calculate_entropy(probs, epsilon=1e-10):
    """
    Calculates entropy of a probability distribution.
    H(p) = - sum(p * log(p))
    """
    # probs shape: (steps, num_classes) or (num_classes,)
    # We want entropy per sample
    return -np.sum(probs * np.log(probs + epsilon), axis=-1)

def main():
    parser = argparse.ArgumentParser(description='Generate predictions and entropy for test images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .h5 model file')
    parser.add_argument('--data_dir', type=str, default='data/test_images', help='Directory containing the test images')
    parser.add_argument('--csv_path', type=str, default='test.csv', help='Path to the test.csv file')
    parser.add_argument('--output_csv', type=str, default='submission_with_entropy.csv', help='Output CSV filename')
    parser.add_argument('--binary', action='store_true', help='Use this flag if the model is binary (2 classes)')
    parser.add_argument('--tta_steps', type=int, default=5, help='Number of TTA steps (1 original + N-1 augmented)')
    
    args = parser.parse_args()
    
    # 1. Load Data
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return
        
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    if 'id_code' not in df.columns:
        print("Error: 'id_code' column not found in CSV.")
        return
        
    # Check if images exist
    print(f"Checking images in {args.data_dir}...")
    valid_indices = []
    image_paths = []
    
    for idx, row in df.iterrows():
        id_code = str(row['id_code'])
        # Try finding the file with common extensions if not provided
        if id_code.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(args.data_dir, id_code)
        else:
            path = os.path.join(args.data_dir, f"{id_code}.png")
            
        if os.path.exists(path):
            valid_indices.append(idx)
            image_paths.append(path)
        else:
            # Try jpg
            path_jpg = os.path.join(args.data_dir, f"{id_code}.jpg")
            if os.path.exists(path_jpg):
                valid_indices.append(idx)
                image_paths.append(path_jpg)
            else:
                print(f"Warning: Image not found for id_code {id_code}")
    
    if not valid_indices:
        print("No valid images found. Exiting.")
        return

    # Filter dataframe to valid images for prediction
    df_valid = df.loc[valid_indices].copy()
    df_valid['image_path'] = image_paths
    
    print(f"Found {len(df_valid)} valid images.")

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []
    
    # 3. Predict Loop
    print(f"Starting prediction with TTA steps={args.tta_steps}...")
    
    for i, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
        img_path = row['image_path']
        original_id = row['id_code']
        
        try:
            # Load and Preprocess
            # Returns numpy array [0, 255]
            img_raw = load_image(img_path)
            
            # Preprocess (returns tensor float32 [0, 255])
            img_tensor = preprocess_image(img_raw)
            
            tta_probs = []
            
            # 1. Original Prediction
            # Expand dims (1, H, W, 3)
            batch = tf.expand_dims(img_tensor, axis=0)
            pred = model.predict(batch, verbose=0)
            tta_probs.append(pred[0])
            
            # 2. Augmented Predictions
            if args.tta_steps > 1:
                for _ in range(args.tta_steps - 1):
                    # Augment
                    aug_tensor = augment_image(img_tensor)
                    batch_aug = tf.expand_dims(aug_tensor, axis=0)
                    pred_aug = model.predict(batch_aug, verbose=0)
                    tta_probs.append(pred_aug[0])
            
            tta_probs = np.array(tta_probs) # shape (steps, num_classes)
            
            # Metrics
            # A. Mean Probability Vector
            mean_probs = np.mean(tta_probs, axis=0)
            
            # B. Predicted Class (Argmax of mean probs)
            predicted_class = np.argmax(mean_probs)
            confidence = mean_probs[predicted_class]
            
            # C. Mean Predicted Entropy
            # Calculate entropy for EACH prediction in TTA, then average
            entropies = calculate_entropy(tta_probs) # shape (steps,)
            mean_predicted_entropy = np.mean(entropies)
            
            results.append({
                'id_code': original_id,
                'target': predicted_class, # Use 'target' or 'diagnosis' to match sample_submission? 
                'probability': confidence,
                'mean_predicted_entropy': mean_predicted_entropy
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Add placeholders or skip? 
            # Better to append None so we have a record
            results.append({
                'id_code': original_id,
                'target': -1,
                'probability': 0.0,
                'mean_predicted_entropy': 0.0
            })

    # 4. Save Results
    results_df = pd.DataFrame(results)
    
    # If binary, maybe reconfirm columns
    # But usually submission requires 'diagnosis' or 'target'
    # sample_submission.csv usually has columns: id_code, diagnosis
    
    # Rename 'target' to 'diagnosis' if that's the standard
    results_df.rename(columns={'target': 'diagnosis'}, inplace=True)
    
    save_path = args.output_csv
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
    print(results_df.head())

if __name__ == "__main__":
    main()
