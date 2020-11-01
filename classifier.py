from typing import Any

import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier(nn.Module):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.bn5 = nn.BatchNorm2d(512)
        self.pooling = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 120)
        self.fc3 = nn.Linear(120, NUM_CLASSES)
        # self.fc = nn.Conv1d(512, NUM_CLASSES, 1)

    def forward(self, x):
        x = self.pooling(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pooling(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pooling(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pooling(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pooling(F.leaky_relu(self.bn5(self.conv5(x))))
        # print(x.size())
        x = x.view(x.size()[0], 512 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
