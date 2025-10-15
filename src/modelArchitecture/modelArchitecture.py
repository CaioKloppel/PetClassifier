import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32), 
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.adjustment1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64)
        )
        self.adjustment2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128)
        )
        self.adjustment3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1)
        )

    def forward(self, x):
        x = self.first_layer(x)

        x = self.pool1(x)
        
        identity = self.adjustment1(x)
        x = self.block1(x)
        x = x + identity 
        x = F.relu(x)

        identity = self.adjustment2(x)
        x = self.block2(x)
        x = x + identity 
        x = F.relu(x)

        identity = self.adjustment3(x)
        x = self.block3(x)
        x = x + identity 
        x = F.relu(x)

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dense_layer(x)
        return x