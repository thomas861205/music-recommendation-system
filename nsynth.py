import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import librosa
import numpy as np
from tqdm import tqdm
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

def wavenet_encode(file_path):
    # Load the model weights.
    checkpoint_path = './wavenet-ckpt/model.ckpt-200000'

    audio, sr = librosa.load(file_path, sr=16000)
    audio = audio[:400000]
    
    # Pass the audio through the first half of the autoencoder,
    # to get a list of latent variables that describe the sound.
    # Note that it would be quicker to pass a batch of audio
    # to fastgen. 
    encoding = fastgen.encode(audio, checkpoint_path, len(audio))
    
    # Reshape to a single sound.
    return encoding.reshape((-1, 16))

splits = {'train': ['bigroom', 'blues', 'classical', 'lofi', 'metal'], 'test': ['country', 'disco', 'dubstep', 'hiphop', 'techno']}
labels = {'bigroom': 0, 'blues': 1, 'classical': 2, 'lofi': 3, 'metal': 4, 'country': 5, 'disco': 6, 'dubstep': 7, 'hiphop': 8, 'techno': 9}

train_feature = []
train_label = []
test_feature = []
test_label = []

for split, genres in splits.items():
    if split == 'train':
        for genre in genres:
            print('Generating Split: {}, Genre: {}'.format(split, genre))
            sound_files = glob.glob('./music/{}/{}/*.mp3'.format(split, genre))
            sound_files = sound_files[:200]

            for sound_file in tqdm(sound_files):
                encoding = wavenet_encode(sound_file)
                label = labels[genre]
                train_feature.append(encoding)
                train_label.append(label)

    else:
        for genre in genres:
            print('Generating Split: {}, Genre: {}'.format(split, genre))
            sound_files = glob.glob('./music/{}/{}/*.mp3'.format(split, genre))
            sound_files = sound_files[:200]

            for sound_file in tqdm(sound_files):
                encoding = wavenet_encode(sound_file)
                label = labels[genre]
                test_feature.append(encoding)
                test_label.append(label)

train_feature = np.array(train_feature)
train_label = np.array(train_label)
np.save('./latent/wavenet/train_feature.npy', train_feature)
np.save('./latent/wavenet/train_label.npy', train_label)
print(train_feature.shape)
print(train_label.shape)
print()

test_feature = np.array(test_feature)
test_label = np.array(test_label)
np.save('./latent/wavenet/test_feature.npy', test_feature)
np.save('./latent/wavenet/test_label.npy', test_label)
print(test_feature.shape)
print(test_label.shape)
