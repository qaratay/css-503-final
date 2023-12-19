import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import soundfile as sf


def text_to_sequence(text, one_hot_encoder):
    text = np.array(list(text)).reshape(-1, 1)
    return one_hot_encoder.transform(text).toarray()


def generate_speech(model, input_text, one_hot_encoder, max_text_length=200):
    input_sequence = text_to_sequence(input_text, one_hot_encoder)
    input_sequence = pad_sequences([input_sequence], maxlen=max_text_length, padding='post')
    predicted_audio = model.predict(input_sequence)
    return predicted_audio[0]


def save_audio_to_file(audio, sample_rate=22050, file_name='output_audio.wav'):
    sf.write(file_name, audio, sample_rate)


model = tf.keras.models.load_model('tts_model.h5')

unique_characters = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                     'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']',
                     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z']

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(np.array(unique_characters).reshape(-1, 1))

input_text = "My first and principal reason was that they enforced beyond all resistance"
audio_output = generate_speech(model, input_text, one_hot_encoder)

save_audio_to_file(audio_output, 22050, 'output_audio.wav')
print("Audio saved as 'output_audio.wav'")
