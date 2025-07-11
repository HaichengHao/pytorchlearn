# model_definition.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net_with_dropout_and_batchnorm(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 卷积层 + 批标准化
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2d = nn.Dropout2d(p=0.5)

        # 全连接层 + BatchNorm
        self.fc1 = nn.Linear(64 * 20 * 20, 1024)
        self.bn1d1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 256)
        self.bn1d2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1d1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn1d2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x