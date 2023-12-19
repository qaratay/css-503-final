import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Masking
from tensorflow.keras.models import Model

# Path and parameters
PATH_TO_DATA = '<FULL_PATH_TO_DATA>'
SAMPLE_RATE = 22050
MAX_TEXT_LENGTH = 200
N_MELS = 80
N_FFT = 2048
HOP_LENGTH = 512

# Preprocessing Functions
def preprocess_audio_to_mel(audio_path, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.T

def text_to_sequence(text, one_hot_encoder):
    text = np.array(list(text)).reshape(-1, 1)
    return one_hot_encoder.transform(text).toarray()

def pad_mel_spectrograms(mel_specs, max_length):
    padded_specs = []
    for spec in mel_specs:
        padded_spec = np.pad(spec, ((0, max_length - spec.shape[0]), (0, 0)), mode='constant')
        padded_specs.append(padded_spec)
    return np.array(padded_specs)

# Model Definition
def define_tts_model(input_dim, output_dim, units=128):
    text_input = Input(shape=(None, input_dim))
    mask = Masking(mask_value=0.0)(text_input)
    x = Bidirectional(LSTM(units, return_sequences=True))(mask)
    x = Bidirectional(LSTM(units))(x)
    audio_output = Dense(output_dim, activation='linear')(x)
    model = Model(inputs=text_input, outputs=audio_output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Loading dataset dictionary
dataset_dict = {}
directories_list = os.listdir(path=PATH_TO_DATA)
directories_list = [dir_name for dir_name in directories_list if dir_name != ".DS_Store"]
for directory in directories_list:
    subdirectories_list = os.listdir(f'{PATH_TO_DATA}/{directory}')
    subdirectories_list = [subdir_name for subdir_name in subdirectories_list if subdir_name != ".DS_Store"]

    for subdirectory in subdirectories_list:
        filenames_list = os.listdir(f'{PATH_TO_DATA}/{directory}/{subdirectory}')
        for filename in filenames_list:
            if 'normalized.txt' in filename and os.path.exists(f'{PATH_TO_DATA}/{directory}/{subdirectory}/{filename[:-15]}.wav'):
                with open(f'{PATH_TO_DATA}/{directory}/{subdirectory}/{filename}') as file:
                    dataset_dict[f'{PATH_TO_DATA}/{directory}/{subdirectory}/{filename[:-15]}.wav'] = file.read().strip()

# Unique characters extraction
unique_characters = sorted(set(''.join(dataset_dict.values())))
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(np.array(unique_characters).reshape(-1, 1))

# Preprocess data
text_sequences = [text_to_sequence(text, one_hot_encoder) for text in dataset_dict.values()]
audio_features = [preprocess_audio_to_mel(path) for path in dataset_dict.keys()]

# Find the maximum length of Mel-spectrograms and pad them
max_mel_length = max(spec.shape[0] for spec in audio_features)
padded_audio_features = pad_mel_spectrograms(audio_features, max_mel_length)

# Padding text sequences
text_sequences = pad_sequences(text_sequences, maxlen=MAX_TEXT_LENGTH, padding='post')

# Reshape padded audio features to match the model's expected output shape
output_dim = max_mel_length * N_MELS
padded_audio_features_flat = np.reshape(padded_audio_features, (len(padded_audio_features), output_dim))

# Defining the Model
input_dim = len(unique_characters)
model = define_tts_model(input_dim, output_dim)

# Loop training
def train_model(model, text_data, padded_audio_data_flat, epochs=10, batch_size=32):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(0, len(text_data), batch_size):
            text_batch = text_data[i:i + batch_size]
            audio_batch = padded_audio_data_flat[i:i + batch_size]
            loss = model.train_on_batch(text_batch, audio_batch)
            print(f"Batch {i // batch_size}, Loss: {loss}")

train_model(model, np.array(text_sequences), padded_audio_features_flat, epochs=10, batch_size=32)

# Saving the model
model.save('tts_model.h5')
