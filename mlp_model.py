import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Simple 2-layer Multi-Layer Perceptron
    Architecture: Input(2) -> Hidden(2) -> Output(1)
    """
    
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        super(SimpleMLP, self).__init__()
        
        # Define layers
        self.hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, output_size, bias=True)
        
        # Store layer information for visualization hooks
        self.activations = {}
        
    def forward(self, x):
        # Forward pass through hidden layer with ReLU activation
        hidden_out = F.relu(self.hidden(x))
        self.activations['hidden'] = hidden_out
        
        # Forward pass through output layer (no activation for regression)
        output = self.output(hidden_out)
        self.activations['output'] = output
        
        return output
    
    def get_weights(self):
        """Return current weights and biases for visualization"""
        return {
            'hidden_weight': self.hidden.weight.data.clone(),
            'hidden_bias': self.hidden.bias.data.clone(),
            'output_weight': self.output.weight.data.clone(),
            'output_bias': self.output.bias.data.clone()
        }
    
    def get_activations(self):
        """Return last computed activations"""
        return self.activations.copy()