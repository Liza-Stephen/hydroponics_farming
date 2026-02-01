"""
GRU (Gated Recurrent Unit) model for time-series forecasting
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class GRUModel(nn.Module):
    """
    GRU model for time-series forecasting of sensor readings
    
    Architecture:
    - GRU layers for sequence learning (lighter than LSTM)
    - Fully connected layers for output
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Initialize GRU model
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in GRU
            num_layers: Number of GRU layers
            output_size: Number of output features to predict
            dropout: Dropout rate
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the last output from the sequence
        gru_out = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(gru_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def predict(self, x, device="cpu"):
        """
        Make predictions
        
        Args:
            x: Input tensor
            device: Device to run on (cpu or cuda)
        
        Returns:
            Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            x = x.to(device)
            predictions = self.forward(x)
            return predictions.cpu().numpy()


def create_gru_model(input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, learning_rate=0.001):
    """
    Create and return GRU model with optimizer and loss function
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_layers: Number of GRU layers
        output_size: Number of output features
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
    
    Returns:
        model, optimizer, criterion
    """
    model = GRUModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    return model, optimizer, criterion
