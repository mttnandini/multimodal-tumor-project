import os
import torch

def count_parameters(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path="models/saved_model.pth"):
    """Save the trained model."""
    torch.save(model.state_dict(), path)
