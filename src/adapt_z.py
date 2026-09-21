import torch
import torch.nn as nn

class AdaptZ(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptZ, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        z = self.fc2(x)
        return z
    