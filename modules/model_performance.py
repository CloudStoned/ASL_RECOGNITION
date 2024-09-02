import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
from datetime import datetime, timedelta

class ModelPerformanceVisualizer:
    def __init__(self, results):
        """
        Initialize the ModelPerformanceVisualizer with results dictionary.

        Args:
        results (dict): dictionary containing lists of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        """
        self.results = results

    def get_preds(self, model, dataloader, device):
        """
        Generate true labels and predictions from a model and a dataloader.

        Args:
        model (torch.nn.Module): Trained model for evaluation.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

        Returns:
        tuple: Two lists containing true labels and predicted labels.
        """
        y_true = []
        y_pred = []

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        return y_true, y_pred

    def plot_loss_curves(self):
        """
        Plots training and validation loss and accuracy curves.
        """
        loss = self.results['train_loss']
        test_loss = self.results['test_loss']
        accuracy = self.results['train_acc']
        test_accuracy = self.results['test_acc']

        epochs = range(len(loss))

        plt.figure(figsize=(15, 7))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='train_loss')
        plt.plot(epochs, test_loss, label='test_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label='train_accuracy')
        plt.plot(epochs, test_accuracy, label='test_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """
        Plots a confusion matrix using actual and predicted labels.

        Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): List of class names
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

    def plot_all(self, y_true, y_pred, classes, save_path=None):
        """
        Plots both loss curves and confusion matrix, and optionally saves the plot.

        Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): List of class names
        save_path (str, optional): Path to save the plot as an image file. If None, the plot is not saved.
        """
        plt.figure(figsize=(20, 15))

        # Plot loss curves
        plt.subplot(2, 2, 1)
        epochs = range(len(self.results['train_loss']))
        plt.plot(epochs, self.results['train_loss'], label='train_loss')
        plt.plot(epochs, self.results['test_loss'], label='test_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        # Plot accuracy curves
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.results['train_acc'], label='train_accuracy')
        plt.plot(epochs, self.results['test_acc'], label='test_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()

        # Plot confusion matrix
        plt.subplot(2, 1, 2)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()

        # Save the plot as a JPG file if save_path is provided
        if save_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format='jpg')
            print(f"Plot saved to {save_path}")

        plt.show()


