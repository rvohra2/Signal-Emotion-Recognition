import librosa
import numpy as np

# Define a class for data augmentation
class DataAugmentation:
    def __init__(self, Ravdess_df):
        # Initialize the class with a Ravdess dataframe
        # Store the path to the audio file
        self.path = np.array(Ravdess_df.Path)[1]
        # Load the audio data with a duration of 2.5 seconds and an offset of 0.6 seconds
        self.data, self.sample_rate = librosa.load(self.path, duration=2.5, offset=0.6)

    # Define a method for adding noise to the data
    def noise(self, data):
        # Calculate a random noise amplitude between 0 and 0.035 times the maximum data value
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        # Add the noise to the data
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        # Return the noisy data
        return data

    # Define a method for stretching the data
    def stretch(self, data, rate=0.8):
        # Stretch the data by the specified rate
        return librosa.effects.time_stretch(data, rate=rate)

    # Define a method for shifting the data
    def shift(self,data):
        # Calculate a random shift range between -5 and 5 milliseconds
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        # Shift the data by the specified range
        return np.roll(data, shift_range)

    # Define a method for pitching the data
    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        # Pitch the data by the specified factor
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

    