import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mlp_model import SimpleMLP


def generate_synthetic_data(n_samples=1000, task='regression'):
    """
    Generate synthetic dataset for training
    
    Args:
        n_samples: Number of samples to generate
        task: 'regression' or 'classification'
    
    Returns:
        X: Input features (n_samples, 2)
        y: Target values (n_samples, 1)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate 2D input features
    X = np.random.randn(n_samples, 2)
    
    if task == 'regression':
        # Simple regression: y = x1^2 + x2^2 + noise
        y = (X[:, 0]**2 + X[:, 1]**2 + 0.1 * np.random.randn(n_samples)).reshape(-1, 1)
    elif task == 'classification':
        # XOR-like problem for binary classification
        y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32).reshape(-1, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)


def train_model(model, X_train, y_train, epochs=1000, lr=0.01, task='regression'):
    """
    Train the MLP model
    
    Args:
        model: SimpleMLP instance
        X_train: Training input features
        y_train: Training targets
        epochs: Number of training epochs
        lr: Learning rate
        task: 'regression' or 'classification'
    
    Returns:
        losses: List of loss values during training
    """
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model and data to device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    # Define loss function and optimizer
    if task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Task: {task}")
    print(f"Learning rate: {lr}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    print("Training completed!")
    return losses


def plot_training_curve(losses):
    """Plot the training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Create model
    model = SimpleMLP()
    
    # Generate training data
    X_train, y_train = generate_synthetic_data(n_samples=1000, task='regression')
    
    print("Dataset Info:")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")
    print(f"Sample input: {X_train[:3]}")
    print(f"Sample output: {y_train[:3]}")
    print()
    
    # Train the model
    losses = train_model(model, X_train, y_train, epochs=1000, lr=0.01, task='regression')
    
    # Plot training curve
    plot_training_curve(losses)
    
    # Test inference
    print("\nTesting inference:")
    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_input = torch.FloatTensor([[1.0, 2.0], [0.5, -1.0], [-1.0, 0.5]]).to(device)
        predictions = model(test_input)
        print(f"Test inputs: {test_input}")
        print(f"Predictions: {predictions}")
        
        # Show activations
        activations = model.get_activations()
        print(f"Hidden activations: {activations['hidden']}")
        
        # Show weights
        weights = model.get_weights()
        print(f"Hidden layer weights:\n{weights['hidden_weight']}")
        print(f"Output layer weights:\n{weights['output_weight']}")