import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

"""
SPLITTED_DATASET = "DATASET"
visualizer = ImageVisualizer(SPLITTED_DATASET)
visualizer.visualize_random_image(seed=30)

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

visualizer.plot_transformed_images(transform, n=2, seed=42)

"""
class ImageVisualizer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG"]
        self.image_path_list = self._get_image_paths()

    def _get_image_paths(self):
        image_path_list = []
        for ext in self.image_extensions:
            image_path_list.extend(self.dataset_path.rglob(ext))
        return image_path_list

    def visualize_random_image(self, seed=None):
        """
        Visualize a random image from the dataset.
        
        Args:
        seed (int, optional): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        if not self.image_path_list:
            print("No images found in the specified path.")
            return
        
        random_image_path = random.choice(self.image_path_list)
        image_class = random_image_path.parent.stem
        
        img = Image.open(random_image_path)
        
        print(f"Random image path: {random_image_path}")
        print(f"Image class: {image_class}")
        print(f"Image height: {img.height}") 
        print(f"Image width: {img.width}")
        
        img_as_array = np.asarray(img)
        
        plt.figure(figsize=(10, 7))
        plt.imshow(img_as_array)
        plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
        plt.axis(False)
        plt.show()

    def plot_transformed_images(self, transform, n=3, seed=None):
        """
        Plots a series of random images from the dataset with their transformed versions.

        Args:
            transform (PyTorch Transforms): Transforms to apply to images.
            n (int, optional): Number of images to plot. Defaults to 3.
            seed (int, optional): Random seed for the random generator. If None, no seed is set.
        """
        if seed is not None:
            random.seed(seed)
        
        random_image_paths = random.sample(self.image_path_list, k=n)
        
        for image_path in random_image_paths:
            with Image.open(image_path) as f:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(f) 
                ax[0].set_title(f"Original \nSize: {f.size}")
                ax[0].axis("off")

                # Transform and plot image
                transformed_image = transform(f)
                if isinstance(transformed_image, torch.Tensor):
                    transformed_image = transformed_image.permute(1, 2, 0).numpy()
                ax[1].imshow(transformed_image) 
                ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
                ax[1].axis("off")

                fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
                plt.tight_layout()
                plt.show()

