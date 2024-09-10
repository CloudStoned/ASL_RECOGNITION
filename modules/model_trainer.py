import torch
import time
import copy
from tqdm.auto import tqdm

class ModelTrainer:
    EARLY_STOPPING_PATIENCE = 10

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"ModelTrainer initialized with device: {self.device}")
        if self.device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")

    def train_step(self, model, dataloader, loss_fn, optimizer):
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def test_step(self, model, dataloader, loss_fn):
        model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

    def train(self, model, train_loader, test_loader, optimizer, loss_fn, epochs, scheduler=None, patience=None):
        print(f"Training on device: {self.device}")
        print(f"Model is on device: {next(model.parameters()).device}")
        
        if next(model.parameters()).device != self.device:
            print(f"Moving model to {self.device}")
            model.to(self.device)
        
        print(f"After moving, model is on device: {next(model.parameters()).device}")
        
        patience = patience if patience is not None else self.EARLY_STOPPING_PATIENCE
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_loss = float('inf')
        best_test_accuracy = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(model, train_loader, loss_fn, optimizer)
            test_loss, test_acc = self.test_step(model, test_loader, loss_fn)
            
            if scheduler:
                scheduler.step(test_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | lr: {current_lr:.6f}")
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f'Early stopping counter: {early_stopping_counter} out of {patience}')
                if early_stopping_counter >= patience:
                    print('Early stopping triggered.')
                    break
        
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - start_time
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return model, results
    

class TensorHandLandmarkTrainer:
    EARLY_STOPPING_PATIENCE = 10

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"TensorHandLandmarkTrainer initialized with device: {self.device}")
        if self.device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")

    def train_step(self, model, X, y, loss_fn, optimizer):
        model.train()
        
        X, y = X.to(self.device), y.to(self.device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc = (y_pred_class == y).sum().item() / len(y)
        
        return loss.item(), acc

    def test_step(self, model, X, y, loss_fn):
        model.eval()
        
        with torch.inference_mode():
            X, y = X.to(self.device), y.to(self.device)
            
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            acc = (test_pred_labels == y).sum().item() / len(y)
            
        return loss.item(), acc

    def train(self, model, X_train, y_train, X_test, y_test, optimizer, loss_fn, epochs, scheduler=None, patience=None, batch_size=32):
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        print(f"Training on device: {self.device}")
        print(f"Model is initially on device: {next(model.parameters()).device}")
        
        try:
            if next(model.parameters()).device != self.device:
                print(f"Attempting to move model to {self.device}")
                model = model.to(self.device)
            
            print(f"After moving, model is on device: {next(model.parameters()).device}")
            
            # Try a forward pass with a small batch
            small_batch = X_train[:1].to(self.device)
            with torch.no_grad():
                output = model(small_batch)
            print("Forward pass successful with small batch")
            
            # Print model structure
            print("Model structure:")
            print(model)
            
            # Print optimizer details
            print("Optimizer details:")
            print(optimizer)
            
            # Print loss function details
            print("Loss function:")
            print(loss_fn)
            
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print("CUDA error details:")
            if torch.cuda.is_available():
                print(torch.cuda.get_device_properties(self.device))
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(self.device) / 1e6:.2f} MB")
                print(f"CUDA memory reserved: {torch.cuda.memory_reserved(self.device) / 1e6:.2f} MB")
            raise  # Re-raise the exception after printing diagnostic info

        patience = patience if patience is not None else self.EARLY_STOPPING_PATIENCE
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_loss = float('inf')
        best_test_accuracy = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = 0, 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                batch_loss, batch_acc = self.train_step(model, batch_X, batch_y, loss_fn, optimizer)
                train_loss += batch_loss
                train_acc += batch_acc
            train_loss /= (len(X_train) // batch_size)
            train_acc /= (len(X_train) // batch_size)
            
            test_loss, test_acc = self.test_step(model, X_test, y_test, loss_fn)
            
            if scheduler:
                scheduler.step(test_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | lr: {current_lr:.6f}")
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f'Early stopping counter: {early_stopping_counter} out of {patience}')
                if early_stopping_counter >= patience:
                    print('Early stopping triggered.')
                    break
        
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - start_time
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return model, results