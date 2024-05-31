import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import librosa.display
from IPython.display import Audio, display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from train import train
from validate import validate
from test import test
import model
import config
from RavdessDataset import RavdessDataset
from torchvision import transforms
import gc

# Clear Python's garbage collector and empty PyTorch's CUDA cache
gc.collect()
torch.cuda.empty_cache()

# Define a PyTorch transform to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Create an instance of the RavdessDataset class, without applying any transforms
dataset = RavdessDataset(directory=config.PATH, transform = None)

# Split dataset into training, validation, and test sets
# 80% for training and validation, 20% for testing
train_val_data, test_data = train_test_split(dataset, test_size=0.2, random_state=0)
# Split the training and validation sets (75% for training, 25% for validation)
train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=0)

# Create DataLoaders for each set, with batch sizes and shuffling as specified
train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)

# Instantiate the EmotionRecognizer model with the specified number of features and output layers
model = model.EmotionRecognizer(config.NUM_FEAT,config.OUTPUT_LAYER)

# Move the model to the specified device (e.g. GPU or CPU)
device = torch.device(config.DEVICE)
model.to(device)

# Define the loss function as Cross-Entropy Loss and the optimizer as Adam with the specified learning rate
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config.LR)

# Train the model for the specified number of epochs
train(model, train_loader, criterion, optimizer, config.NUM_EPOCHS)
# Validate the model on the validation set
validate(model, val_loader, criterion)
# Test the model on the test set
test(model, test_loader, criterion)






