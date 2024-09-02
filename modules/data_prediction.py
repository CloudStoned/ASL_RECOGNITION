import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from torchvision import datasets

"""
Usage example:
predictor = ComprehensiveImagePredictor(model, class_names, transform)

# Predict and plot a single image from file
predictor.predict_and_plot_image("path/to/your/image.jpg")

# Predict and visualize multiple images from a dataset
test_data = YourTestDataset()  # Your test dataset
predictor.predict_and_visualize_multiple(test_data, num_samples=9, seed=42)
"""


class ComprehensiveImagePredictor:
    def __init__(self, model: torch.nn.Module, class_names: list, transform=None, device=None):
        """
        Initialize the ComprehensiveImagePredictor.

        Args:
        model (torch.nn.Module): The trained model for making predictions.
        class_names (list): List of class names.
        transform (callable, optional): Transformations to apply to the images.
        device (str, optional): The device to run the model on. If None, it will be automatically determined.
        """
        self.model = model
        self.class_names = class_names
        self.transform = transform
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def predict_and_plot_image(self, image_path: str):
        """
        Makes a prediction on a target image and plots the image with its prediction.

        Args:
        image_path (str): Path to the image file.
        """
        # Load and transform image
        target_image = Image.open(image_path).convert("RGB")
        if self.transform:
            target_image = self.transform(target_image)
        
        # Prepare image for prediction
        target_image = target_image.unsqueeze(dim=0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            target_image_pred = self.model(target_image)
        
        # Process prediction
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        
        # Prepare image for plotting
        target_image = target_image.squeeze().cpu().numpy()
        target_image = np.transpose(target_image, (1, 2, 0))
        
        # Plot
        plt.imshow(target_image)
        title = f"Pred: {self.class_names[target_image_pred_label.item()]} | Prob: {target_image_pred_probs.max().item():.3f}"
        plt.title(title)
        plt.axis('off')
        plt.show()

    def predict_and_visualize_multiple(self, model, directory, transformer, num_samples, seed=None):

        if seed is not None:
            random.seed(seed)

        dataset = datasets.ImageFolder(root=directory, transform=transformer)

        # Randomly select test samples and labels
        random_indices = random.sample(range(len(dataset)), k=num_samples)  # Adjust k as needed for the number of samples
        test_samples = [dataset[i][0] for i in random_indices]  # Extract the samples
        test_labels = [dataset[i][1] for i in random_indices]  # Extract the labels

        # Make predictions
        model.eval()
        pred_classes = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with torch.no_grad():
            for sample in test_samples:
                sample = torch.unsqueeze(sample, dim=0).to(device)  # Prepare sample
                pred_logit = model(sample)  # Forward pass
                pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)  # Get prediction probability
                pred_class = torch.argmax(pred_prob).item()  # Get predicted class
                pred_classes.append(pred_class)

        # Plot predictions
        plt.figure(figsize=(9, 9))
        nrows = 3
        ncols = 3
        for i, sample in enumerate(test_samples):
            # Create a subplot
            plt.subplot(nrows, ncols, i + 1)

            # Plot the target image
            plt.imshow(sample.squeeze().permute(1, 2, 0))  # Adjust channel order if necessary

            # Find the prediction label (in text form, e.g., "Sandal")
            pred_label = dataset.classes[pred_classes[i]]

            # Get the truth label (in text form, e.g., "T-shirt")
            truth_label = dataset.classes[test_labels[i]]

            # Create the title text of the plot
            title_text = f"Pred: {pred_label} | Truth: {truth_label}"

            # Check for equality and change title colour accordingly
            if pred_label == truth_label:
                plt.title(title_text, fontsize=10, color="g")  # Green text if correct
            else:
                plt.title(title_text, fontsize=10, color="r")  # Red text if wrong

            plt.axis('off')

        plt.tight_layout()
        plt.show()

