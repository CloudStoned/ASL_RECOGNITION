import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
Example usage:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
creator = DataLoaderCreator(train_dir="path/to/train_dir",
                            test_dir="path/to/test_dir",
                            transform=transform)
                            
train_dataloader, test_dataloader, class_names = creator.create_dataloaders(batch_size=32)
print(f"Class names: {creator.get_class_names()}")

"""

class DataLoaderCreator:
    def __init__(self, train_dir: str, test_dir: str, transform: transforms.Compose):
        """
        Initialize the DataLoaderCreator with directories and transform.

        Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (transforms.Compose): torchvision transforms to perform on training and testing data.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform
        self.train_data = None
        self.test_data = None
        self.class_names = None
        self.num_workers = os.cpu_count()

    def _create_datasets(self):
        """Create ImageFolder datasets for training and testing data."""
        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.transform)
        self.class_names = self.train_data.classes

    def create_dataloaders(self, batch_size: int, num_workers: int = None):
        """
        Creates training and testing DataLoaders.

        Args:
        batch_size (int): Number of samples per batch in each of the DataLoaders.
        num_workers (int, optional): An integer for number of workers per DataLoader.
                                     If None, uses the number of CPU cores.

        Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.

        Example usage:
        creator = DataLoaderCreator(train_dir="path/to/train_dir",
                                    test_dir="path/to/test_dir",
                                    transform=some_transform)
        train_dataloader, test_dataloader, class_names = creator.create_dataloaders(batch_size=32)
        """
        if self.train_data is None or self.test_data is None:
            self._create_datasets()

        if num_workers is not None:
            self.num_workers = num_workers

        train_dataloader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, self.class_names

    def get_class_names(self):
        """
        Returns the class names.

        Returns:
        A list of class names.
        """
        if self.class_names is None:
            self._create_datasets()
        return self.class_names

