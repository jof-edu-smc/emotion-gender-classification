import os
import torch # type: ignore
import torchaudio # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
import librosa  # type: ignore
import subprocess
from tqdm import tqdm   # type: ignore
import vggish_params

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, label_type='emotion'):
        """
        Per Kaggle Page: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
        Filename identifiers
            - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
            - Vocal channel (01 = speech, 02 = song).
            - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
            - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
            - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
            - Repetition (01 = 1st repetition, 02 = 2nd repetition).
            - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

        Filename example: 03-01-06-01-02-01-12.wav
            - Audio-only (03)
            - Speech (01)
            - Fearful (06)
            - Normal intensity (01)
            - Statement "dogs" (02)
            - 1st Repetition (01)
            - 12th Actor (12)
            - Female, as the actor ID number is even.
        """
        if label_type == 'emotion':
            self.label_type = self._create_emotion_label(self.file_paths)
        elif label_type == 'gender':
            self.label_type = self._create_gender_label(self.file_paths)
        else:
            raise ValueError("Invalid label type. Must be 'emotion' or 'gender'.")
        
        self.data_dir = data_dir
        self.label_type = label_type
        self.file_paths = self._index_data(data_dir)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=vggish_params.SAMPLE_RATE)
        
        self.emotion_map = {
            1: 'neutral',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fearful',
            7: 'disgust',
            8: 'surprised'
        }
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=vggish_params.SAMPLE_RATE,
            n_mels=vggish_params.NUM_MEL_BINS,
            n_fft=int(vggish_params.STFT_WINDOW_LENGTH_SECONDS * vggish_params.SAMPLE_RATE),
            hop_length=int(vggish_params.STFT_HOP_LENGTH_SECONDS * vggish_params.SAMPLE_RATE),
            f_min=vggish_params.MEL_MIN_HZ,
            f_max=vggish_params.MEL_MAX_HZ
        )
        
       
    
    def average_wav_file_length(self):
        total_length = 0
        for file_path in tqdm(self.file_paths, desc="Calculating average length"):
            try:
                audio, fs = torchaudio.load(file_path)
                total_length += audio.shape[1] / fs  # Length in seconds
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        average_length = total_length / len(self.file_paths) if self.file_paths else 0
        return average_length
    
    def idx_to_emotion(self, code):
        return self.emotion_map.get(code, 'unknown')
    
    def _index_data(self, data_dir):
        # Recursively index all .wav files in the data directory
        file_paths = []
        for root, _, files in os.walk(data_dir):
            if './ravdess/Actor_' in root:  # Only consider directories that match the expected format
                for file in files:
                    if file.endswith('.wav'):
                        file_paths.append(os.path.join(root, file))
        print(f"Indexed {len(file_paths)} audio files from {data_dir}")
        
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("Duplicate file paths detected in the dataset directory.")
        return file_paths
    
    def _create_emotion_label(self, file_paths):
        # Extract emotion labels from file paths
        labels = []
        for path in file_paths:
            # RAVDESS file format: "Actor_01/03-01-01-01-01-01-01.wav"
            emotion_code = int(path.split("-")[2])  # Extract the emotion code
            labels.append(emotion_code - 1)  # Subtract 1 to make it zero-indexed
            
        return labels
    
    def _create_gender_label(self, file_paths):
        # Extract emotion labels from file paths
        labels = []
        for path in file_paths:
            # RAVDESS file format: "Actor_01/03-01-01-01-01-01-01.wav"
            actor_code = int(path.split("-")[-1].split(".")[0]) # Extract the emotion code
            if actor_code % 2 == 0:
                # Is Female 
                labels.append(1)
            else:
                labels.append(0)
        return torch.tensor(labels, dtype=torch.long)
    
    def _mel_spectrogram_examples(self, audio, fs):
        """Return exactly MAX_EXAMPLES log-mel patches [MAX_EXAMPLES, 1, 96, 64]."""
        MAX_EXAMPLES = 4  # Set this to a fixed number that fits your data
        
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)

        if audio.ndim > 1:
            audio = torch.mean(audio, dim=0)
        audio = audio.to(torch.float32)

        # 1. Generate the Mel Spectrogram
        mel_spec = self.mel_transform(audio)

        log_offset = getattr(vggish_params, 'LOG_OFFSET', 0.01)
        log_mel = torch.log(mel_spec + log_offset)
        log_mel = log_mel.transpose(0, 1).contiguous()  # [Total_Frames, 64]

        # 2. Calculate Required Frames for MAX_EXAMPLES
        features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
        window_len = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate)) # 96
        hop_len = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))      # 96

        # Target frames = (Number of windows - 1) * hop + window_length
        required_frames = (MAX_EXAMPLES - 1) * hop_len + window_len
        total_frames, num_bands = log_mel.shape

        # 3. Pad or Truncate to reach exactly required_frames
        if total_frames < required_frames:
            # Pad with zeros if the audio is too short
            pad_amount = required_frames - total_frames
            log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, pad_amount), mode='constant', value=0.0)
        else:
            # Truncate if the audio is too long
            log_mel = log_mel[:required_frames, :]

        # 4. Slice into exactly MAX_EXAMPLES
        examples = []
        for i in range(MAX_EXAMPLES):
            start = i * hop_len
            end = start + window_len
            examples.append(log_mel[start:end, :])

        # Stack to [MAX_EXAMPLES, 1, 96, 64]
        stacked = torch.stack(examples, dim=0).to(torch.float32)
        return stacked.unsqueeze(1)

    def _load_audio(self, file_path):
        try: 
            audio, fs = torchaudio.load(file_path)
        except Exception as e:
            # print(f"Error loading {file_path} with torchaudio: {e}. Falling back to librosa.")
            audio, fs = librosa.load(file_path, sr=None, mono=False)
            audio = torch.tensor(audio, dtype=torch.float32)
        if fs != vggish_params.SAMPLE_RATE:
            # print(f"Resampling {file_path} from {fs} Hz to {vggish_params.SAMPLE_RATE} Hz.")
            audio = self.resampler(audio)
            fs = vggish_params.SAMPLE_RATE
        samples = audio / 32768.0  # Normalize to [-1.0, +1.0] range based on max int16 value
        return samples, fs
        
    def __len__(self):
        """Return the total number of audio files in the dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio, fs = self._load_audio(self.file_paths[idx])
        mel_spec = self._mel_spectrogram_examples(audio, fs)
        # Return the mel spectrogram and the corresponding label (emotion or gender)
        return mel_spec, self.label_type[idx]
    
def download_ravdess():
    # Download and extract RAVDESS dataset
    subprocess.run([
        "curl", "-L", "-o", "./ravdess-emotional-speech-audio.zip",
        "https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-speech-audio"
    ], check=True)
    subprocess.run(
        ["unzip", "-o", "./ravdess-emotional-speech-audio.zip", "-d", "./ravdess"], 
        check=True
    )
    