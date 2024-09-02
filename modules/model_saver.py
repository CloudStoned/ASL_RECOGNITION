from pathlib import Path
import torch

"""
# Example usage:
# saver = ModelSaver(target_dir="models")
# saver.save_model(model=model_0, model_name="05_going_modular_tingvgg_model.pth")
# loaded_model = saver.load_model(model=new_model_instance, model_name="05_going_modular_tingvgg_model.pth")

"""

class ModelSaver:
    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: torch.nn.Module, model_name: str):

        # Check file extension
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

        # Create model save path
        model_save_path = self.target_dir / model_name

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    def load_model(self, 
                   model: torch.nn.Module, 
                   model_name: str):
        model_path = self.target_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        print(f"[INFO] Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
        return model

