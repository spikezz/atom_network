import torch
from mlp_model import SimpleMLP

# Check device availability
print("Device Information:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Create model and check device
model = SimpleMLP()
print(f"\nModel device: {next(model.parameters()).device}")

# Create tensor and check device
x = torch.randn(10, 2)
print(f"Tensor device: {x.device}")

# Show how to move to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    model_gpu = model.to(device)
    x_gpu = x.to(device)
    
    print(f"\nAfter moving to GPU:")
    print(f"Model device: {next(model_gpu.parameters()).device}")
    print(f"Tensor device: {x_gpu.device}")
    
    # Test forward pass on GPU
    output = model_gpu(x_gpu)
    print(f"Output device: {output.device}")
else:
    print("\nGPU not available, using CPU")