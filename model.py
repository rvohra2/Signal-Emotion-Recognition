import torch.nn as nn

class EmotionRecognizer(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmotionRecognizer, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x