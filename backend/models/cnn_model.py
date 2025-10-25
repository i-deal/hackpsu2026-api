import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
from typing import List, Tuple, Dict
import random

class ViolenceCNN(nn.Module):
    
    def __init__(self, num_classes=2):
        super(ViolenceCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pre_recurrence = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        self.recurrence = nn.Sequential(
            nn.Linear(1024, 800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(800, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, batch):
        y = torch.zeros_like(batch[0])

        for i in range(len(x)):
            x = batch[i]
            x = self.features(x)
            x = self.pre_recurrence(x)
            z = torch.cat([y,x], 0)
            y = self.recurrence(z)
            
        x = self.classifier(y)
        return x