
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from config import get_config
from utils.logger import logger

config = get_config()

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel:
    def __init__(self, input_size, sequence_length=60):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.hidden_size = 50
        self.num_layers = 2
        self.num_epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        
        self.model = LSTMNet(input_size, self.hidden_size, self.num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"Initialized LSTM Model on {self.device}")

    def create_sequences(self, data, targets=None):
        """
        Convert 2D array to 3D sequences
        """
        xs = []
        ys = []
        
        # Convert DataFrame to numpy if needed
        if hasattr(data, 'values'):
            data = data.values
        if targets is not None and hasattr(targets, 'values'):
            targets = targets.values
            
        for i in range(len(data) - self.sequence_length):
            x = data[i:(i + self.sequence_length)]
            xs.append(x)
            if targets is not None:
                # Target is the value AFTER the sequence
                ys.append(targets[i + self.sequence_length])
                
        return np.array(xs), np.array(ys) if targets is not None else None

    def train(self, X_train, y_train):
        self.model.train()
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_train, y_train)
        
        if len(X_seq) == 0:
            logger.warning("Not enough data for LSTM sequences")
            return
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 10 == 0:
                logger.debug(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
                
        logger.info("LSTM training complete")

    def predict(self, X):
        self.model.eval()
        
        # For prediction, we need the last sequence
        if len(X) < self.sequence_length:
            logger.warning("Not enough data for LSTM prediction")
            return None
            
        # Create simple sequence (just the last window)
        # Using the last 'sequence_length' rows
        last_sequence = X.iloc[-self.sequence_length:].values
        
        X_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(X_tensor)
            
        return prediction.cpu().numpy()[0][0]

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved LSTM model to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded LSTM model from {path}")
        else:
            logger.warning(f"No model found at {path}")
