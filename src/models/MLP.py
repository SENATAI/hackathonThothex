import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[800, 300, 150], output_dim=8, dropout=0.3):
        """
        input_dim: размер входного вектора
        hidden_dims: список размеров скрытых слоев, например [800, 300, 150]
        output_dim: количество классов
        dropout: dropout между слоями
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Инициализация весов
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # уменьшает размер вдвое
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 61 * 61, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

