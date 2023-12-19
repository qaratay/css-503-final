# Text-to-Speech Model

This project contains a deep learning model for text-to-speech (TTS) conversion using TensorFlow and Librosa. The model takes text as input and generates speech audio. The code is structured into two main files: `tts.py` for training the model and `model_usage.py` for using the trained model to generate speech from text.

## Project Structure

- `tts.py`: This script includes code for loading and preprocessing the dataset, defining the TTS model architecture, training the model, and saving the trained model.
- `model_usage.py`: This script is used to load the trained model, convert input text to speech, and save the generated speech as an audio file.

## Setup and Running

### Dependencies

To run the scripts, you need Python and the following libraries:
- TensorFlow
- NumPy
- Librosa
- Scikit-learn
- SoundFile

You can install these dependencies via pip:
```bash
pip install tensorflow numpy librosa scikit-learn soundfile
```

LibriSpeech ASR corpus was downloaded from here (test-clean was downloaded)
https://www.openslr.org/12
