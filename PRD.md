# Product Requirements Document: Simple 2-Layer MLP

## Project Overview
Build the simplest possible Multi-Layer Perceptron (MLP) neural network using PyTorch with a well-defined, minimal architecture.

## Technical Requirements

### Network Architecture
- **Input Layer**: 2 neurons (2 input features)
- **Hidden Layer**: 2 neurons with activation function
- **Output Layer**: 1 neuron (single output prediction)
- **Total Layers**: 2 trainable layers (hidden + output)

### Implementation Details
- **Framework**: PyTorch
- **Activation Function**: ReLU for hidden layer, no activation for output (regression) or Sigmoid (binary classification)
- **Weight Initialization**: PyTorch default initialization
- **Bias**: Include bias terms in both layers

### Network Structure
```
Input (2) → Hidden Layer (2) → Output (1)
```

### Functional Requirements

#### Core Components
1. **Model Definition**
   - Define MLP class inheriting from `nn.Module`
   - Two linear layers: `nn.Linear(2, 2)` and `nn.Linear(2, 1)`
   - Forward pass implementation

2. **Training Setup**
   - Loss function (MSE for regression or BCELoss for classification)
   - Optimizer (SGD or Adam)
   - Training loop with forward/backward pass

3. **Data Handling**
   - Simple synthetic dataset generation
   - Data preprocessing if needed
   - Train/validation split (optional for demo)

4. **Demonstration**
   - Training the model on sample data
   - Inference example with new input
   - Basic performance visualization

5. **Real-time Visualization System**
   - Interactive network diagram showing neuron activations
   - Real-time monitoring of tensor values during training/inference
   - Live updates of weights and biases
   - Activation flow visualization through the network

### Non-Functional Requirements
- **Simplicity**: Minimal code with clear structure
- **Readability**: Well-commented code explaining each step
- **Modularity**: Separate concerns (model, training, data)

## Implementation Plan

### Phase 1: Core Implementation
1. Set up PyTorch imports and dependencies
2. Define the MLP model class
3. Create simple synthetic dataset
4. Implement training loop
5. Add inference capability

### Phase 2: Visualization System
1. Implement real-time network visualization
2. Add tensor flow monitoring hooks
3. Create interactive dashboard for training observation

### Phase 3: Enhancement
1. Add basic visualization (loss curve)
2. Include model evaluation metrics
3. Add example usage and documentation

## Success Criteria

### Core Functionality
- [ ] Model trains without errors
- [ ] Forward pass produces correct output shape (batch_size, 1)
- [ ] Loss decreases during training
- [ ] Model can make predictions on new data
- [ ] Code is clean and well-documented

### Visualization System
- [ ] Real-time display of neuron activations during forward pass
- [ ] Live monitoring of weight/bias values during training
- [ ] Interactive network diagram with clickable neurons
- [ ] Tensor value inspection at each layer
- [ ] Smooth animation of data flow through network
- [ ] Training progress visualization with activation heatmaps

## File Structure
```
atom_network/
├── mlp_model.py          # MLP class definition with visualization hooks
├── train.py              # Training script with real-time monitoring
├── demo.py               # Usage demonstration
├── visualizer.py         # Real-time network visualization system
├── dashboard.py          # Interactive training dashboard
├── hooks.py              # PyTorch forward/backward hooks for tensor monitoring
└── requirements.txt      # Dependencies
```

## Dependencies
- torch
- numpy (for data generation)
- matplotlib (for visualization)
- plotly (for interactive plots)
- dash (for web-based dashboard)
- threading (for concurrent visualization updates)

## Detailed Visualization Requirements

### Real-time Network Monitoring
1. **Neuron Activation Display**
   - Color-coded neurons showing activation intensity
   - Numerical values displayed on hover/click
   - Smooth transitions between forward passes

2. **Tensor Flow Visualization**
   - Animated arrows showing data flow direction
   - Thickness/color intensity representing magnitude
   - Step-by-step progression through layers

3. **Weight/Bias Monitoring**
   - Heatmaps for weight matrices
   - Real-time updates during training
   - Historical tracking of parameter changes

4. **Interactive Features**
   - Pause/resume training for detailed inspection
   - Click neurons to see detailed tensor information
   - Adjust visualization speed and detail level
   - Export snapshots of network state

### Technical Implementation Details
- **PyTorch Hooks**: Use forward hooks to capture intermediate activations
- **Threading**: Separate visualization thread to avoid blocking training
- **Data Structure**: Efficient storage of activation history for playback
- **Performance**: Minimal overhead on training performance

## Example Use Cases
1. **Binary Classification**: Predict 0/1 based on 2 input features
2. **Regression**: Predict continuous value from 2 inputs
3. **Learning XOR**: Classic non-linearly separable problem
4. **Visualization Demo**: Watch network learn patterns in real-time
5. **Educational Tool**: Understand neural network internals through visualization