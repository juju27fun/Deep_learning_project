import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import IMG_SIZE, NUM_CLASSES, BATCH_SIZE, SEED, SPLIT_RATIO, EPOCHS_PHASE_1, EPOCHS_PHASE_2
from src.preprocessing import load_image, preprocess_image, augment_image
from src.models import build_model
from src.evaluation import calculate_qwk, plot_confusion_matrix, get_classification_report, QWKCallback
from src.interpretability import visualize_gradcam

def create_dataset(dataframe, is_training=True):
    paths = dataframe['id_code'].values
    labels = dataframe['diagnosis'].astype(int).values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def load_and_preprocess(path, label):
        # Use tf.numpy_function to wrap the OpenCV loading
        # This allows us to use complex OpenCV logic (like Ben Graham's cropping) in the TF pipeline
        [img] = tf.numpy_function(load_image, [path], [tf.uint8])
        
        # Ensure shape is known (numpy_function loses shape info)
        img.set_shape([None, None, 3])
        
        img, label = preprocess_image(img, label)
        return img, label
    
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(dataframe))
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def plot_history(h1, h2, qwk_history, save_path):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 6))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    # QWK
    plt.subplot(1, 3, 3)
    plt.plot(range(len(qwk_history)), qwk_history, label='Validation QWK')
    plt.legend(loc='lower right')
    plt.title('Validation QWK Score')
    
    plt.savefig(save_path)
    plt.close()

def tta_predict(model, dataset, steps=5):
    """
    Test Time Augmentation (TTA).
    Predicts multiple times on augmented versions of the image and averages the results.
    """
    print(f"Running TTA with {steps} steps...")
    final_preds = []
    true_labels = []
    
    # We need to iterate carefully. 
    # dataset is (images, labels). We want to augment images 'steps' times.
    
    # Extract all images and labels first (might be memory intensive for huge datasets, 
    # but for this size it's okay. Alternatively, iterate batch by batch)
    
    for images, labels in dataset:
        batch_preds = np.zeros((images.shape[0], NUM_CLASSES))
        
        # Original prediction
        batch_preds += model.predict(images, verbose=0)
        
        # Augmented predictions
        for _ in range(steps - 1):
            # Apply augmentation manually
            aug_images = augment_image(images)
            batch_preds += model.predict(aug_images, verbose=0)
            
        # Average
        batch_preds /= steps
        final_preds.extend(batch_preds)
        true_labels.extend(labels.numpy())
        
    return np.array(true_labels), np.array(final_preds)

def main():
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    # 1. Load and Split Data
    print("Loading data...")
    df = pd.read_csv('train.csv')
    df['id_code'] = df['id_code'].apply(lambda x: f"data/train_images/{x}.png")
    df['diagnosis'] = df['diagnosis'].astype(str)

    train_df, temp_df = train_test_split(df, test_size=(1 - SPLIT_RATIO[0]), random_state=SEED, stratify=df['diagnosis'])
    val_df, test_df = train_test_split(temp_df, test_size=(SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2])), random_state=SEED, stratify=temp_df['diagnosis'])

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Calculate Class Weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['diagnosis'].astype(int)),
        y=train_df['diagnosis'].astype(int)
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # Create Datasets
    train_ds = create_dataset(train_df, is_training=True)
    val_ds = create_dataset(val_df, is_training=False)
    test_ds = create_dataset(test_df, is_training=False)

    # 2. Instantiate Model
    print("Building model...")
    model = build_model(learning_rate=1e-3, freeze_backbone=True)
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('outputs/best_model.h5', monitor='val_accuracy', save_best_only=True)
    qwk_callback = QWKCallback(val_ds)

    # 3. Phase 1 Training (Frozen Backbone)
    print(f"Starting Phase 1 Training ({EPOCHS_PHASE_1} epochs)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE_1,
        callbacks=[early_stopping, reduce_lr, checkpoint, qwk_callback],
        class_weight=class_weights_dict
    )

    # 4. Phase 2 Training (Fine-tuning)
    print(f"Starting Phase 2 Training ({EPOCHS_PHASE_2} epochs)...")
    model.trainable = True
    
    # Use Label Smoothing in Loss for Phase 2
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) 
    # Note: SparseCategoricalCrossentropy doesn't support label_smoothing directly in older TF versions 
    # or requires one-hot encoding. 
    # To keep it simple and compatible, we stick to standard loss but rely on TTA and Preprocessing for QWK boost.
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=['accuracy']
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE_2,
        callbacks=[early_stopping, reduce_lr, checkpoint, qwk_callback],
        class_weight=class_weights_dict
    )

    # 6. Plot Training History
    plot_history(history_phase1, history_phase2, qwk_callback.history, 'outputs/training_history.png')

    # 7. Final Evaluation with TTA
    print("Evaluating on Test Set (with TTA)...")
    best_model = tf.keras.models.load_model('outputs/best_model.h5')
    
    # Use TTA for prediction
    y_true, y_pred_probs = tta_predict(best_model, test_ds, steps=3)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate QWK
    qwk = calculate_qwk(y_true, y_pred)
    print(f"Final Quadratic Weighted Kappa (QWK) with TTA: {qwk:.4f}")

    # Classification Report
    classes = [str(i) for i in range(NUM_CLASSES)]
    print(get_classification_report(y_true, y_pred, classes))

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, classes, save_path='outputs/confusion_matrix.png')

    # 8. Grad-CAM Visualization (on non-augmented images)
    print("Generating Grad-CAM visualizations...")
    
    # Re-create test_ds to get raw images easily for visualization
    # (TTA loop consumed the iterator)
    test_ds_vis = create_dataset(test_df, is_training=False)
    
    test_images = []
    test_labels = []
    for img, lbl in test_ds_vis.take(1): # Take one batch
        test_images = img.numpy()
        test_labels = lbl.numpy()
        break
        
    if len(test_images) > 0:
        indices = np.random.choice(len(test_images), min(5, len(test_images)), replace=False)
        
        for i, idx in enumerate(indices):
            img = test_images[idx]
            true_label = test_labels[idx]
            
            # Get prediction for this specific image
            pred_probs = best_model.predict(np.expand_dims(img, axis=0), verbose=0)
            pred_label = np.argmax(pred_probs)
            
            print(f"Sample {i+1}: True Class: {true_label}, Predicted Class: {pred_label}")
            visualize_gradcam(best_model, img, pred_label, save_path=f'outputs/gradcam_sample_{i+1}.png')

if __name__ == "__main__":
    main()