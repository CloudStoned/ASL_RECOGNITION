from torchinfo import summary
import torch.nn as nn


"""
Example usage:
model_summarizer = ModelSummarizer(model=ASL_mobilenet, class_names_count=35)
model_summarizer.print_summary()
"""

class ModelSummarizer:
    def __init__(self, model: nn.Module, class_names_count: int):
        """
        Initialize the ModelSummarizer with a model and the number of classes.

        Args:
        model (torch.nn.Module): The model to summarize.
        class_names_count (int): The number of classes (used in the input size).
        """
        self.model = model
        self.class_names_count = class_names_count

    def print_summary(self):
        """
        Prints the summary of the model.
        """
        summary(model=self.model, 
                input_size=(self.class_names_count, 3, 224, 224),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])


