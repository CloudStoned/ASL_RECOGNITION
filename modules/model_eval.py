import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


"""
Example usage:
model = YourModel()  # Your trained model
test_dataloader = DataLoader(...)  # Your test dataloader
class_names = ["class1", "class2", ...]  # Your class names

evaluator = ModelEvaluator(model)
results = evaluator.evaluate_and_report(test_dataloader, class_names)

print(f"Accuracy: {results['accuracy']:.4f}")
print("Classification Report:")
print(results['classification_report'])

"""

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device=None):
        """
        Initialize the ModelEvaluator.

        Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device, optional): The device to run the model on. If None, it will be automatically determined.
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate_model(self, dataloader: DataLoader):
        """
        Evaluate the model on the given dataloader.

        Args:
        dataloader (DataLoader): The dataloader containing the evaluation data.

        Returns:
        tuple: Predicted labels and true labels.
        """
        self.model.eval()  # Set model to evaluation mode
        all_preds = []
        all_targets = []
        
        with torch.no_grad():  # No need to compute gradients during evaluation
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_logit = self.model(X)
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                
                all_preds.extend(y_pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_targets)

    def compute_accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model predictions.

        Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

        Returns:
        float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    def generate_classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate a classification report.

        Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        target_names (list, optional): List of target class names.

        Returns:
        str: Classification report as a string.
        """
        return classification_report(y_true, y_pred, target_names=target_names)

    def evaluate_and_report(self, dataloader: DataLoader, class_names=None):
        """
        Evaluate the model and generate a complete report.

        Args:
        dataloader (DataLoader): The dataloader containing the evaluation data.
        class_names (list, optional): List of class names for the classification report.

        Returns:
        dict: A dictionary containing accuracy and classification report.
        """
        y_pred, y_true = self.evaluate_model(dataloader)
        
        accuracy = self.compute_accuracy(y_true, y_pred)
        report = self.generate_classification_report(y_true, y_pred, target_names=class_names)
        
        return {
            "accuracy": accuracy,
            "classification_report": report
        }



class HandLandMarksEvaluator:
    def __init__(self, model: torch.nn.Module, device=None):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate_model(self, X, y):
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(X, DataLoader):
                all_preds = []
                all_targets = []
                for batch_X, batch_y in X:
                    batch_X = batch_X.to(self.device)
                    y_logit = self.model(batch_X)
                    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                    
                    all_preds.extend(y_pred.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                return np.array(all_preds), np.array(all_targets)
            else:
                X = X.to(self.device)
                y_logit = self.model(X)
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                
                return y_pred.cpu().numpy(), y.cpu().numpy()

    def compute_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def generate_classification_report(self, y_true, y_pred, target_names=None):
        return classification_report(y_true, y_pred, target_names=target_names)

    def evaluate_and_report(self, X, y, class_names=None):
        y_pred, y_true = self.evaluate_model(X, y)
        
        accuracy = self.compute_accuracy(y_true, y_pred)
        report = self.generate_classification_report(y_true, y_pred, target_names=class_names)
        
        return {
            "accuracy": accuracy,
            "classification_report": report
        }