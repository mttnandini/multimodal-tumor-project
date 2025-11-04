import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_dim1=128, input_dim2=128, num_classes=9):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
