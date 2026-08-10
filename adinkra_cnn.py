import torch.nn as nn

class AdinkraCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdinkraCNN, self).__init__()

        # Block 1 3 to 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2 32 to 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64 x 56 x 56
        )

        # Block 3: 64 to 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128 x 28 x 28
        )

        # Block 4: 128 to 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 256 x 14 x 14
        )

        # Block 5: 256 -> 512 channels
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 512 x 7 x 7
        )

        # Global Average Pooling: 512 x 7 x 7 -> 512 x 1 x 1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3), # Drops 30% of connections to prevent memorization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten from (Batch, 512, 1, 1) to (Batch, 512)

        x = self.classifier(x)
        return x



