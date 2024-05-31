import os
from data_aug import DataAugmentation as da
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from torchvision import transforms

class RavdessDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.file_emotion = []
        self.file_path = []
        self.get_files()

    def get_files(self):
        for dir in os.listdir(self.directory):
            actor = os.listdir(os.path.join(self.directory, dir))
            for file in actor:
                part = file.split('.')[0]
                part = part.split('-')
                self.file_emotion.append(int(part[2]))
                self.file_path.append(os.path.join(self.directory, dir, file))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        path = self.file_path[idx]
        emotion = self.file_emotion[idx]
        feature = self.get_features(path)
        #feature = transforms.ToPILImage()(torch.from_numpy(feature))
        feature = torch.tensor(feature, dtype=torch.float32)
        #emotion = torch.tensor(emotion, dtype=torch.float32)
        if self.transform:
            feature = self.transform(feature)
        return feature, emotion
    
    def extract_features(self, data, sample_rate):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        #Energy
        # energy = np.array([
        # sum(abs(data[i:i+1024]**2))
        # for i in range(0, len(data), 512)
        # ])
        # result=np.hstack((result, energy))

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally
        
        return result

    def get_features(self, path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        # without augmentation
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)
        
        # data with noise
        noise_data = da.noise(self,data)
        res2 = self.extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        new_data = da.stretch(self,data)
        data_stretch_pitch = da.pitch(self,new_data, sample_rate)
        res3 = self.extract_features(data_stretch_pitch, sample_rate)
        result = np.vstack((result, res3)) # stacking vertically
        return result