import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
            
        score = calculate_qwk(y_true, y_pred)
        self.history.append(score)
        print(f" - val_qwk: {score:.4f}")