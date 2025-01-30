from src.models import MLModel
from src.utils.config_loader import load_config

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()  # Initialize the nn.Module base class
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Define forward pass."""
        x = torch.relu(self.fc1(x))  
        x = self.dropout1(x)  
        x = torch.relu(self.fc2(x))  
        x = torch.relu(self.fc3(x)) 
        x = self.fc4(x)  
        x = torch.relu(x)  #  non-negative outputs
        return x
    
class NeuralNet(MLModel):
    def __init__(self, input_dim, **kwargs):
        super().__init__(name="NeuralNet")
        
        # Load configuration from YAML file
        config = load_config(self.config_path)
            
        default_params = config.get('NeuralNet', {})
        self.params = {**default_params, **kwargs}
        
        # Set input dimension based on data
        self.params["input_dim"] = input_dim
        self.params["lr"] = float(self.params["lr"])
        self.scaler = StandardScaler()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model parameters from configuration
        input_dim = self.params["input_dim"]
        hidden_dim = self.params["hidden_dim"]
        dropout_rate = self.params["dropout_rate"]
        output_dim = self.params["output_dim"]
        
        self.model = MLP(input_dim, hidden_dim, output_dim, dropout_rate)  # Create the model container
        
        # Initialize loss function, optimizer, and early stopping parameters
        self.criterion = self._get_loss_function()
        self.optimizer = self._get_optimizer()
        self.patience = self.params.get("patience", 10)
        self.best_valid_loss = float("inf")
        self.epochs_without_improvement = 0
        
        # Move model to device
        self.model = self.model.to(self.device)

    def _get_loss_function(self):
        """Get the loss function."""
        loss_name = self.params.get("loss", "L1Loss")
        if loss_name == "L1Loss":
            return nn.L1Loss()
        elif loss_name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _get_optimizer(self):
        """Get the optimizer."""
        optimizer_name = self.params.get("optimizer", "Adam")
        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.params["lr"])
        elif optimizer_name == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.params["lr"])
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _fit_impl(self, X, y, X_val=None, y_val=None):
        # Convert data to PyTorch tensors
        X = self.scaler.fit_transform(X)
        X_val = self.scaler.transform(X_val)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.params["batch_size"], shuffle=False)
        else:
            val_dataloader = None

        # Training loop
        for epoch in range(self.params["epochs"]):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.params['epochs']}",
                                unit="batch", dynamic_ncols=True, leave=False)

            for batch_X, batch_y in progress_bar:
                self.optimizer.zero_grad()
                outputs = self.model.forward(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # Update the progress bar description
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
                
            
            self.logger.info(f"Epoch [{epoch+1}/{self.params['epochs']}], Train Loss: {running_loss/len(dataloader):.4f}")
            
            # Validation (if validation data is provided)
            if val_dataloader is not None:
                self.model.eval()  # Set model to evaluation mode
                valid_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        outputs = self.model.forward(batch_X).squeeze()
                        valid_loss += self.criterion(outputs, batch_y).item()

                # Average validation loss
                valid_loss /= len(val_dataloader)
                if (epoch + 1) % self.params["print_per_epoch"] == 0 or (epoch + 1) == self.params["epochs"]:
                    self.logger.info(f"Epoch [{epoch+1}/{self.params['epochs']}], Valid Loss: {valid_loss:.4f}")


                # Early stopping
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.epochs_without_improvement = 0
                    self.save_model()
                    
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                self.logger.info(f"Epoch [{epoch+1}/{self.params['epochs']}], Train Loss: {running_loss/len(dataloader):.4f}")

    def _predict_impl(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = self.model.forward(X_tensor).cpu().numpy()
        return predictions

