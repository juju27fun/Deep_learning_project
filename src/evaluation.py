import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

def calculate_qwk(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa score.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        
    Returns:
        float: The QWK score.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Generates and plots the confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_classification_report(y_true, y_pred, classes):
    """
    Calculates per-class Precision, Recall, and F1-score.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
        
    Returns:
        str: The classification report.
    """
    return classification_report(y_true, y_pred, target_names=classes)

class QWKCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to calculate QWK on validation set at the end of each epoch.
    """
    def __init__(self, validation_data):
        super(QWKCallback, self).__init__()
        self.validation_data = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []
        
        # Iterate over the validation dataset
        for images, labels in self.validation_data:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            
        score = calculate_qwk(y_true, y_pred)
        self.history.append(score)
        print(f" - val_qwk: {score:.4f}")

class PerClassMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to calculate Precision, Recall, and F1 for each class 
    on the validation set at the end of each epoch.
    """
    def __init__(self, validation_data, num_classes=5):
        super(PerClassMetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes
        # Dictionary to store history: {'f1_0': [], 'rec_0': [], ..., 'f1_4': []}
        self.history = {} 

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import precision_recall_fscore_support
        
        y_true = []
        y_pred = []
        
        # Predict on validation data
        # We need raw probabilities for Loss calculation
        all_preds = [] 
        all_labels = []

        for images, labels in self.validation_data:
            preds = self.model.predict(images, verbose=0)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Convert to class indices for metric calculation
        y_true = np.argmax(all_labels, axis=1)
        y_pred = np.argmax(all_preds, axis=1)
            
        # Calculate metrics per class
        # strict labels list ensures we get results for all classes even if some are missing in batch
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.num_classes)), zero_division=0
        )
        
        # Calculate additional per-class metrics
        for i in range(self.num_classes):
            # 1. Precision, Recall, F1 (Standard)
            self.history.setdefault(f'precision_{i}', []).append(precision[i])
            self.history.setdefault(f'recall_{i}', []).append(recall[i])
            self.history.setdefault(f'f1_{i}', []).append(f1[i])

            # 2. Per-Class Accuracy (One-vs-Rest)
            # True Positive: Predicted i, True i
            # True Negative: Predicted not-i, True not-i
            # Accuracy = (TP + TN) / Total
            binary_y_true = (y_true == i).astype(int)
            binary_y_pred = (y_pred == i).astype(int)
            acc = np.mean(binary_y_true == binary_y_pred)
            self.history.setdefault(f'accuracy_{i}', []).append(acc)

            # 3. Per-Class QWK (treated as Binary Kappa for One-vs-Rest)
            # Since QWK on binary data is equivalent to Cohen's Kappa
            k = cohen_kappa_score(binary_y_true, binary_y_pred)
            # Handle NaN if class is missing
            if np.isnan(k): k = 0.0
            self.history.setdefault(f'qwk_{i}', []).append(k)

            # 4. Per-Class Loss
            # Categorical Crossentropy on samples that truly belong to class i
            # Filter samples where true label is i
            class_indices = np.where(y_true == i)[0]
            if len(class_indices) > 0:
                # Get probabilities for these samples
                class_probs = all_preds[class_indices]
                # Get target (one-hot) for these samples
                class_targets = all_labels[class_indices]
                
                # Calculate CrossEntropy
                cce = tf.keras.losses.CategoricalCrossentropy()
                loss = cce(class_targets, class_probs).numpy()
            else:
                loss = 0.0
            
            self.history.setdefault(f'loss_{i}', []).append(loss)
            
        print(f" - val_mean_f1: {np.mean(f1):.4f}")

def save_predictions(ids, y_true, y_pred_probs, save_path):
    """
    Saves predictions to a CSV file.
    
    Args:
        ids (list): List of image IDs (filenames).
        y_true (array-like): True labels.
        y_pred_probs (array-like): Predicted probabilities (N, num_classes).
        save_path (str): Path to save the CSV.
    """
    import pandas as pd
    
    df = pd.DataFrame()
    # Clean up IDs if they are full paths
    df['id_code'] = [os.path.basename(i) for i in ids] 
    df['diagnosis'] = y_true
    df['predicted_diagnosis'] = np.argmax(y_pred_probs, axis=1)
    
    # Add probabilities
    for i in range(y_pred_probs.shape[1]):
        df[f'prob_{i}'] = y_pred_probs[:, i]
        
    df.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}")