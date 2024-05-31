# Signal-Emotion-Recognition
Emotion recognition using audio signals

## Introduction
This repository contains code and resources for recognizing emotions from audio signals using machine learning techniques. The goal of this project is to develop a system that can accurately identify emotions such as happy, sad, anger, neutral, calm, disgust, surprise, and fear from both video and audio recordings.

## Features
    Dataset: The dataset used here is Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess).
    Augmentation: The code includes functions for preprocessing audio signals, such as noise injection, time shifting, frequency shift, pitch change and speed.
    Preprocessing: For preprocessing, we performed feature extraction using techniques such as: zero-cross rating, entropy of energy, root mean square value, chroma shift, and Mel Frequency Cepstral Coefficients.
    Model Training: The repository includes implementations of convolutional neural networks (CNN).
    Evaluation: The code includes functions for evaluating the performance of the models using metrics such as accuracy.

## Usage
    Clone the repository: git clone (https://github.com/rvohra2/Signal-Emotion-Recognition.git)
    Install the required libraries: pip install -r requirements.txt
    Run the model: speechEmotionAnalysis.py

## Results
The results show that the CNN model achieved the highest accuracy of 90% on the test set.

## Future Work
    Improving the model: Experiment with different architectures and hyperparameters to improve the accuracy of the model.
    Expanding the dataset: Collect more audio and video recordings to increase the size and diversity of the dataset.
    Real-time implementation: Develop a real-time system that can recognize emotions from audio signals in real-time.
