#!/usr/bin/env python3

import sys
import torch
import torch.nn as nn

# Test loading the MLP model safely
def test_mlp_loading():
    try:
        print("Loading model file...")
        checkpoint = torch.load('models/mlp/mlp.pth', map_location='cpu', weights_only=False)
        
        print("Model file loaded successfully")
        print("Keys:", list(checkpoint.keys()))
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("State dict keys:", list(state_dict.keys()))
            
            # Check layer shapes
            for key, value in state_dict.items():
                if key.startswith('pred.'):
                    print(f"{key}: {value.shape}")
                    
            # Try creating a simple model
            class TestMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.pred = nn.Sequential(
                        nn.Linear(4116, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1)
                    )
                    
                def forward(self, x):
                    return self.pred(x)
            
            model = TestMLP()
            
            # Create filtered state dict
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('pred.'):
                    new_key = key.replace('pred.', '')
                    filtered_state_dict[new_key] = value
                    print(f"Mapping {key} -> {new_key}")
            
            # Try loading
            model.pred.load_state_dict(filtered_state_dict)
            print("Model loaded successfully!")
            
            # Test forward pass
            test_input = torch.randn(1, 4116)
            output = model(test_input)
            print(f"Test output shape: {output.shape}")
            print(f"Test output value: {output.item()}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mlp_loading()
    print(f"Test {'PASSED' if success else 'FAILED'}") 