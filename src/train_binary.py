import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import IMG_SIZE, BATCH_SIZE, SEED, SPLIT_RATIO, EPOCHS_PHASE_1, EPOCHS_PHASE_2
from src.preprocessing import load_image, preprocess_image, augment_image
from src.models import build_model
from src.evaluation import calculate_qwk, plot_confusion_matrix, get_classification_report, QWKCallback, PerClassMetricsCallback, save_predictions
from src.interpretability import visualize_gradcam

OUTPUT_DIR = 'outputs_binary'

# Override NUM_CLASSES for binary classification
NUM_CLASSES = 2

def create_dataset(dataframe, is_training=True):
    paths = dataframe['id_code'].values
    labels = dataframe['diagnosis'].astype(int).values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def load_and_preprocess(path, label):
        # Use tf.numpy_function to wrap the OpenCV loading
        [img] = tf.numpy_function(load_image, [path], [tf.uint8])
        
        # Ensure shape is known (numpy_function loses shape info)
        img.set_shape([None, None, 3])
        
        img, label = preprocess_image(img, label)
        label = tf.one_hot(label, NUM_CLASSES)
        return img, label
    
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(dataframe))
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def plot_history(h1, h2, qwk_history, per_class_history=None, save_dir=OUTPUT_DIR):
    # Combine history from both phases
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    precision = h1.history['precision'] + h2.history['precision']
    val_precision = h1.history['val_precision'] + h2.history['val_precision']
    recall = h1.history['recall'] + h2.history['recall']
    val_recall = h1.history['val_recall'] + h2.history['val_recall']

    # Calculate F1 Score
    f1 = []
    for p, r in zip(precision, recall):
        if p + r > 0:
            f1.append(2 * p * r / (p + r))
        else:
            f1.append(0)
            
    val_f1 = []
    for p, r in zip(val_precision, val_recall):
        if p + r > 0:
            val_f1.append(2 * p * r / (p + r))
        else:
            val_f1.append(0)

    epochs_range = range(len(acc))

    # Helper function to plot and save
    def save_plot(metric_train, metric_val, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, metric_train, label=f'Training {ylabel}')
        plt.plot(epochs_range, metric_val, label=f'Validation {ylabel}')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    # 1. Accuracy
    save_plot(acc, val_acc, 'Training and Validation Accuracy', 'Accuracy', 'plot_accuracy.png')

    # 2. Loss
    save_plot(loss, val_loss, 'Training and Validation Loss', 'Loss', 'plot_loss.png')
    
    # 3. Precision
    save_plot(precision, val_precision, 'Training and Validation Precision', 'Precision', 'plot_precision.png')
    
    # 4. Recall
    save_plot(recall, val_recall, 'Training and Validation Recall', 'Recall', 'plot_recall.png')
    
    # 5. F1 Score
    save_plot(f1, val_f1, 'Training and Validation F1 Score', 'F1 Score', 'plot_f1.png')
    
    # 6. QWK (Validation only)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(qwk_history)), qwk_history, label='Validation QWK')
    plt.legend(loc='lower right')
    plt.title('Validation Quadratic Weighted Kappa')
    plt.xlabel('Epochs')
    plt.ylabel('QWK')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plot_qwk.png'))
    plt.close()

    # 7. Per-Class Metrics (if provided)
    if per_class_history:
        classes = sorted(list(set(int(k.split('_')[1]) for k in per_class_history.keys() if 'precision' in k)))
        
        def plot_per_class(metric_name, title, filename):
            plt.figure(figsize=(10, 6))
            for c in classes:
                key = f'{metric_name}_{c}'
                if key in per_class_history:
                    plt.plot(range(len(per_class_history[key])), per_class_history[key], label=f'Class {c}')
            
            plt.legend(loc='lower right')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel(metric_name.capitalize())
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()

        plot_per_class('precision', 'Validation Precision per Class', 'per_class_precision.png')
        plot_per_class('recall', 'Validation Recall per Class', 'per_class_recall.png')
        plot_per_class('f1', 'Validation F1 Score per Class', 'per_class_f1.png')
        plot_per_class('accuracy', 'Validation Accuracy per Class (One-vs-Rest)', 'per_class_accuracy.png')
        plot_per_class('loss', 'Validation Loss per Class', 'per_class_loss.png')
        plot_per_class('qwk', 'Validation QWK per Class (Binary)', 'per_class_qwk.png')

def tta_predict(model, dataset, steps=5):
    """
    Test Time Augmentation (TTA).
    """
    print(f"Running TTA with {steps} steps...")
    final_preds = []
    true_labels = []
    
    for images, labels in dataset:
        batch_preds = np.zeros((images.shape[0], NUM_CLASSES))
        
        # Original prediction
        batch_preds += model.predict(images, verbose=0)
        
        # Augmented predictions
        for _ in range(steps - 1):
            aug_images = augment_image(images)
            batch_preds += model.predict(aug_images, verbose=0)
            
        # Average
        batch_preds /= steps
        final_preds.extend(batch_preds)
        true_labels.extend(labels.numpy())
        
    return np.array(true_labels), np.array(final_preds)

def plot_data_distribution(df, save_dir=OUTPUT_DIR):
    plt.figure(figsize=(8, 6))
    counts = df['diagnosis'].value_counts().sort_index()
    counts.plot(kind='bar', color='skyblue')
    plt.title('Data Distribution per Class (Binary)')
    plt.xlabel('Diagnosis Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(counts):
        plt.text(i, v + 5, str(v), ha='center', fontsize=10)
        
    plt.savefig(os.path.join(save_dir, 'data_distribution.png'))
    plt.close()

def save_class_samples(df, save_dir=OUTPUT_DIR, samples_per_class=5):
    print("Generating sample images per class...")
    
    unique_classes = df['diagnosis'].unique()
    unique_classes.sort()
    
    for cls in unique_classes:
        class_df = df[df['diagnosis'] == cls]
        samples = class_df.sample(min(samples_per_class, len(class_df)), replace=False)
        
        plt.figure(figsize=(15, 3))
        plt.suptitle(f'Sample Images - Class {cls}', fontsize=16)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            img_path = row['id_code'] # Already formatted as path in main
            
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                
                plt.subplot(1, samples_per_class, i + 1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"{os.path.basename(img_path)}")
            else:
                print(f"Warning: Image not found {img_path}")
                
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_class_{cls}.png'))
        plt.close()

def main():
    # Ensure outputs directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and Split Data
    print("Loading data...")
    df = pd.read_csv('train.csv')
    df['id_code'] = df['id_code'].apply(lambda x: f"data/train_images/{x}.png")
    
    # MAP TO BINARY
    # 0 -> 0 (No Event)
    # 1, 2, 3, 4 -> 1 (Event)
    print("Mapping to Binary Labels (0=No Event, 1=Event)...")
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
    df['diagnosis'] = df['diagnosis'].astype(int)

    train_df, temp_df = train_test_split(df, test_size=(1 - SPLIT_RATIO[0]), random_state=SEED, stratify=df['diagnosis'])
    val_df, test_df = train_test_split(temp_df, test_size=(SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2])), random_state=SEED, stratify=temp_df['diagnosis'])

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Calculate Class Weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['diagnosis']),
        y=train_df['diagnosis']
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # Create Datasets
    train_ds = create_dataset(train_df, is_training=True)
    val_ds = create_dataset(val_df, is_training=False)
    test_ds = create_dataset(test_df, is_training=False)

    # 2. Instantiate Model
    print("Building model for 2 classes...")
    model = build_model(learning_rate=1e-3, freeze_backbone=True, num_classes=NUM_CLASSES)
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'), monitor='val_accuracy', save_best_only=True)
    
    # Note: QWKCallback might assume continuous ordinal classes, but here we have binary. 
    # It should still work as "kappa" or we can just ignore it if it's not relevant. 
    # But QWK is often used for this specific dataset type (diabetic retinopathy) even if binary? 
    # Typically QWK is for ordinal. For binary, it's just Kappa.
    qwk_callback = QWKCallback(val_ds) 
    pc_callback = PerClassMetricsCallback(val_ds, NUM_CLASSES)

    # 3. Phase 1 Training (Frozen Backbone)
    print(f"Starting Phase 1 Training ({EPOCHS_PHASE_1} epochs)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE_1,
        callbacks=[early_stopping, reduce_lr, checkpoint, qwk_callback, pc_callback],
        class_weight=class_weights_dict
    )

    # ================= FROZEN BACKBONE TRAINING STOPS HERE =================
    
    # 4. Phase 2 Training (Fine-tuning)
    print(f"Starting Phase 2 Training ({EPOCHS_PHASE_2} epochs)...")
    model.trainable = True
    
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE_2,
        callbacks=[early_stopping, reduce_lr, checkpoint, qwk_callback, pc_callback],
        class_weight=class_weights_dict
    )

    # 6. Plot Training History
    plot_history(history_phase1, history_phase2, qwk_callback.history, pc_callback.history, OUTPUT_DIR)

    # 7. Final Evaluation with TTA
    print("Evaluating on Test Set (with TTA)...")
    best_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'best_model.h5'))
    
    y_true, y_pred_probs = tta_predict(best_model, test_ds, steps=3)
    y_true = np.argmax(y_true, axis=1) 
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate QWK (Binary Kappa)
    qwk = calculate_qwk(y_true, y_pred)
    print(f"Final Quadratic Weighted Kappa (QWK) with TTA: {qwk:.4f}")

    # Classification Report
    classes = [str(i) for i in range(NUM_CLASSES)]
    print(get_classification_report(y_true, y_pred, classes))

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, classes, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

    # 8. Grad-CAM Visualization
    print("Generating Grad-CAM visualizations...")
    test_ds_vis = create_dataset(test_df, is_training=False)
    
    test_images = []
    test_labels = []
    for img, lbl in test_ds_vis.take(1):
        test_images = img.numpy()
        test_labels = lbl.numpy()
        break
        
    if len(test_images) > 0:
        indices = np.random.choice(len(test_images), min(5, len(test_images)), replace=False)
        for i, idx in enumerate(indices):
            img = test_images[idx]
            true_label = np.argmax(test_labels[idx])
            pred_probs = best_model.predict(np.expand_dims(img, axis=0), verbose=0)
            pred_label = np.argmax(pred_probs)
            print(f"Sample {i+1}: True Class: {true_label}, Predicted Class: {pred_label}")
            visualize_gradcam(best_model, img, pred_label, save_path=os.path.join(OUTPUT_DIR, f'gradcam_sample_{i+1}.png'))

    # 9. Additional Visualizations
    print("Generating post-training visualizations...")
    plot_data_distribution(df, OUTPUT_DIR)
    save_class_samples(df, OUTPUT_DIR)
    
    # 10. Generate Predictions CSVs
    print("Generating CSV predictions for all sets...")
    
    def get_preds(dataset):
        all_preds = []
        all_labels = []
        for img, lbl in dataset:
            all_preds.extend(best_model.predict(img, verbose=0))
            all_labels.extend(lbl.numpy())
        return np.argmax(np.array(all_labels), axis=1), np.array(all_preds)

    print(" - Predicting on Train set...")
    train_ds_pred = create_dataset(train_df, is_training=False) 
    y_true_train, y_probs_train = get_preds(train_ds_pred)
    save_predictions(train_df['id_code'].values, y_true_train, y_probs_train, os.path.join(OUTPUT_DIR, 'train_predictions.csv'))
    
    print(" - Predicting on Validation set...")
    y_true_val, y_probs_val = get_preds(val_ds)
    save_predictions(val_df['id_code'].values, y_true_val, y_probs_val, os.path.join(OUTPUT_DIR, 'val_predictions.csv'))
    
    print(" - Saving Test set predictions...")
    save_predictions(test_df['id_code'].values, y_true, y_pred_probs, os.path.join(OUTPUT_DIR, 'test_predictions.csv'))

if __name__ == "__main__":
    main()
