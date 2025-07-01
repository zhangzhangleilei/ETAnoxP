import torch.nn as nn
import torch
class FullModel(nn.Module):
    def __init__(self, in_dim):
    # def __init__(self, in_dim, model):
        super(FullModel, self).__init__()
        # self.model = model
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, data):
        x = torch.relu(self.fc1(data))
        x = torch.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc4(x))
        return x