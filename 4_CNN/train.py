import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from PIL import Image
import zipfile
import shutil

# 训练参数的配置
class Training_Parameters:
    RSCD_DIR = "E:\mastercode\DeepLearningForPycharm\4_CNN\data\RSCD"
    TRAIN_PART1 = os.path.join(RSCD_DIR , "train-part1")
    TRAIN_PART2 = os.path.join(RSCD_DIR , "train-part2")
    VAL_DIR = os.path.join(RSCD_DIR , "vali_20k")
    TEST_DIR = os.path.join(RSCD_DIR , "test_50k")

    BATCH_SIZE = 100
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

