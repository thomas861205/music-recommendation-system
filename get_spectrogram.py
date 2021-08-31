import warnings
warnings.filterwarnings('ignore')

import os
import glob
import pickle
import librosa
import numpy as np
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt

splits = {'train': ['bigroom', 'blues', 'classical', 'lofi', 'metal'], 'test': ['country', 'disco', 'dubstep', 'hiphop', 'techno']}
labels = {'bigroom': 0, 'blues': 1, 'classical': 2, 'lofi': 3, 'metal': 4, 'country': 5, 'disco': 6, 'dubstep': 7, 'hiphop': 8, 'techno': 9}

train_spectrogram = []
train_label = []
val_spectrogram = []
val_label = []
test_spectrogram = []
test_label = []

for split, genres in splits.items():
    if split == 'train':
        continue
        for genre in genres:
            print('Generating Split: {}, Genre: {}'.format(split, genre))
            sound_files = glob.glob('./music/{}/{}/*.mp3'.format(split, genre))

            train_sound_files = sound_files[:200]
            val_sound_files = sound_files[200:]

            for train_sound_file in tqdm(train_sound_files):
                y, sr = librosa.load(train_sound_file)
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
                if melspectrogram.shape[1] > 1292:
                    melspectrogram = melspectrogram[:, :1292]
                label = labels[genre]
                train_spectrogram.append(melspectrogram)
                train_label.append(label)

            for val_sound_file in tqdm(val_sound_files):
                y, sr = librosa.load(val_sound_file)
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
                if melspectrogram.shape[1] > 1292:
                    melspectrogram = melspectrogram[:, :1292]
                label = labels[genre]
                val_spectrogram.append(melspectrogram)
                val_label.append(label)

    else:
        for genre in genres:
            print('Generating Split: {}, Genre: {}'.format(split, genre))
            sound_files = glob.glob('./music/{}/{}/*.mp3'.format(split, genre))
            sound_files = sound_files[:200]

            for sound_file in tqdm(sound_files):
                y, sr = librosa.load(sound_file)
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
                if melspectrogram.shape[1] > 1292:
                    melspectrogram = melspectrogram[:, :1292]
                label = labels[genre]
                test_spectrogram.append(melspectrogram)
                test_label.append(label)

# train_spectrogram = np.array(train_spectrogram)
# train_label = np.array(train_label)
# randomize = np.arange(train_spectrogram.shape[0])
# np.random.shuffle(randomize)
# train_spectrogram = train_spectrogram[randomize]
# train_label = train_label[randomize]
# np.save('./spectrogram/train/train_spectrogram.npy', train_spectrogram)
# np.save('./spectrogram/train/train_label.npy', train_label)
# print(train_spectrogram.shape)
# print(train_label.shape)
# print()

# val_spectrogram = np.array(val_spectrogram)
# val_label = np.array(val_label)
# randomize = np.arange(val_spectrogram.shape[0])
# np.random.shuffle(randomize)
# val_spectrogram = val_spectrogram[randomize]
# val_label = val_label[randomize]
# np.save('./spectrogram/val/val_spectrogram.npy', val_spectrogram)
# np.save('./spectrogram/val/val_label.npy', val_label)
# print(val_spectrogram.shape)
# print(val_label.shape)
# print()

test_spectrogram = np.array(test_spectrogram)
test_label = np.array(test_label)
randomize = np.arange(test_spectrogram.shape[0])
np.random.shuffle(randomize)
test_spectrogram = test_spectrogram[randomize]
test_label = test_label[randomize]
np.save('./spectrogram/test/test_spectrogram.npy', test_spectrogram)
np.save('./spectrogram/test/test_label.npy', test_label)
print(test_spectrogram.shape)
print(test_label.shape)

# feature_train = {}
# feature_test = {}
# for i in range(len(genre)):
#     genre_index = i
#     music_dir = './music/' + genre[genre_index]
#     sound_files = [data for data in os.listdir(music_dir) if data.endswith(".mp3")]

#     train_sound_files = sound_files[:200]
#     test_sound_files = sound_files[200:250]

#     ##########################################################################################################################################
#     train_sound_names = [data.split(".mp3")[0] for data in train_sound_files]
#     lines = [line.rstrip('\n') for line in open('./music/{}/feature.txt'.format(genre[genre_index]))]
#     for i in range(len(lines)):
#         temp = lines[i].split(',')
#         song_name = ",".join(temp[:-3])
#         energy = int(float(temp[-3]) * 10)
#         dance = int(float(temp[-2]) * 10)
#         lyric = int(float(temp[-1]) * 10)

#         if song_name in train_sound_names:
#             feature_train[genre[genre_index] + "_" + song_name] = (genre_index, energy, dance, lyric)

#     for i in tqdm(range(len(train_sound_files))):
#         y, sr = librosa.load(music_dir + "/" + train_sound_files[i])
#         melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#         np.save("./spectrogram_train/" + genre[genre_index] + "_" + train_sound_files[i].split(".mp3")[0] + ".npy", melspectrogram)
#     ##########################################################################################################################################

#     ##########################################################################################################################################
#     test_sound_names = [data.split(".mp3")[0] for data in test_sound_files]
#     lines = [line.rstrip('\n') for line in open('./music/{}/feature.txt'.format(genre[genre_index]))]

#     for i in range(len(lines)):
#         temp = lines[i].split(',')
#         song_name = ",".join(temp[:-3])
#         energy = int(float(temp[-3]) * 10)
#         dance = int(float(temp[-2]) * 10)
#         lyric = int(float(temp[-1]) * 10)

#         if song_name in test_sound_names:
#             feature_test[genre[genre_index] + "_" + song_name] = (genre_index, energy, dance, lyric)

#     for i in tqdm(range(len(test_sound_files))):
#         y, sr = librosa.load(music_dir + "/" + test_sound_files[i])
#         melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#         np.save("./spectrogram_test/" + genre[genre_index] + "_" + test_sound_files[i].split(".mp3")[0] + ".npy", melspectrogram)
#     ##########################################################################################################################################

# print(len(feature_train))
# pickle_out = open("feature_train.pickle", "wb")
# pickle.dump(feature_train, pickle_out)
# pickle_out.close()

# print(len(feature_test))
# pickle_out = open("feature_test.pickle", "wb")
# pickle.dump(feature_test, pickle_out)
# pickle_out.close()

# feature_dict = {}
# lines = [line.rstrip('\n') for line in open('./music/{}/feature.txt'.format(genre[genre_index]))]
# for i in range(len(lines)):
#     temp = lines[i].split(',')
#     song_name = temp[0]
#     energy = int(float(temp[1]) * 10)
#     dance = int(float(temp[2]) * 10)
#     lyric = int(float(temp[3]) * 10)
#     feature_dict[song_name] = (genre_index, energy, dance, lyric)

# for i in range(len(sound_files)):
#     song_name = sound_files[i].split('.mp3')[0]
#     print(feature_dict[song_name])

# sound_files = [data for data in os.listdir("./music/hiphop") if data.endswith(".mp3")]
# y, sr = librosa.load("./music/hiphop" + "/" + sound_files[10])
# melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# print(melspectrogram.shape)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()

# print(melspectrogram)
