import torch
import numpy as np
from mlp_model import SimpleMLP
from train import generate_synthetic_data, train_model


def demonstrate_regression():
    """Demonstrate MLP on regression task"""
    print("=" * 60)
    print("REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Create model and data
    model = SimpleMLP()
    X_train, y_train = generate_synthetic_data(n_samples=500, task='regression')
    
    # Train model
    losses = train_model(model, X_train, y_train, epochs=500, lr=0.01, task='regression')
    
    # Test on new data
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor([
            [1.0, 1.0],   # Should predict ~2.0
            [0.0, 0.0],   # Should predict ~0.0
            [2.0, -1.0],  # Should predict ~5.0
            [-1.5, 1.5]   # Should predict ~4.5
        ])
        
        predictions = model(test_inputs)
        
        print(f"\nRegression Results:")
        print(f"Input -> Prediction (Expected: x1² + x2²)")
        for i, (inp, pred) in enumerate(zip(test_inputs, predictions)):
            expected = inp[0]**2 + inp[1]**2
            print(f"[{inp[0]:5.1f}, {inp[1]:5.1f}] -> {pred.item():6.3f} (Expected: {expected:6.3f})")


def demonstrate_classification():
    """Demonstrate MLP on classification task"""
    print("\n" + "=" * 60)
    print("CLASSIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Create model and data
    model = SimpleMLP()
    X_train, y_train = generate_synthetic_data(n_samples=500, task='classification')
    
    # Train model
    losses = train_model(model, X_train, y_train, epochs=1000, lr=0.01, task='classification')
    
    # Test on new data
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor([
            [1.0, 1.0],    # Should predict 1 (positive * positive)
            [1.0, -1.0],   # Should predict 0 (positive * negative)
            [-1.0, 1.0],   # Should predict 0 (negative * positive)
            [-1.0, -1.0],  # Should predict 1 (negative * negative)
            [2.0, 3.0],    # Should predict 1 (both positive)
            [-2.0, 1.5]    # Should predict 0 (mixed signs)
        ])
        
        raw_outputs = model(test_inputs)
        probabilities = torch.sigmoid(raw_outputs)
        predictions = (probabilities > 0.5).float()
        
        print(f"\nClassification Results:")
        print(f"Input -> Probability -> Prediction (Expected: x1*x2 > 0)")
        for i, (inp, prob, pred) in enumerate(zip(test_inputs, probabilities, predictions)):
            expected = 1.0 if (inp[0] * inp[1]) > 0 else 0.0
            print(f"[{inp[0]:5.1f}, {inp[1]:5.1f}] -> {prob.item():5.3f} -> {pred.item():1.0f} (Expected: {expected:1.0f})")


def demonstrate_network_internals():
    """Demonstrate network internal states"""
    print("\n" + "=" * 60)
    print("NETWORK INTERNALS DEMONSTRATION")
    print("=" * 60)
    
    # Create and train a simple model
    model = SimpleMLP()
    X_train, y_train = generate_synthetic_data(n_samples=200, task='regression')
    train_model(model, X_train, y_train, epochs=300, lr=0.01, task='regression')
    
    # Analyze network internals
    model.eval()
    with torch.no_grad():
        test_input = torch.FloatTensor([[1.5, -0.5]])
        output = model(test_input)
        
        print(f"\nNetwork Analysis for input: {test_input[0].tolist()}")
        print("-" * 40)
        
        # Show weights and biases
        weights = model.get_weights()
        print("Hidden Layer Weights:")
        print(f"  W1: {weights['hidden_weight'].tolist()}")
        print(f"  b1: {weights['hidden_bias'].tolist()}")
        
        print("\nOutput Layer Weights:")
        print(f"  W2: {weights['output_weight'].tolist()}")
        print(f"  b2: {weights['output_bias'].tolist()}")
        
        # Show activations
        activations = model.get_activations()
        print(f"\nActivations:")
        print(f"  Hidden layer: {activations['hidden'][0].tolist()}")
        print(f"  Output layer: {activations['output'][0].tolist()}")
        
        print(f"\nFinal prediction: {output.item():.6f}")


def run_all_demonstrations():
    """Run all demonstrations"""
    print("Simple MLP Neural Network Demonstrations")
    print("PyTorch implementation with 2-layer architecture")
    
    demonstrate_regression()
    demonstrate_classification()
    demonstrate_network_internals()
    
    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demonstrations()