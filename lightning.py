import os
import glob
import subprocess
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
import torchaudio
import gc
from pretrained_vggish import vggish_params
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, MulticlassConfusionMatrix, ConfusionMatrix, Precision, F1Score, Recall

class ValidationMetricsCallback(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.detach().cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.detach().cpu().item())
            
class TrainMetricsCallback(Callback):
    def __init__(self):
        self.train_accs_gender = []
        self.train_accs_emotion = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        gender_acc = trainer.callback_metrics.get("train_gender_acc")
        emotion_acc = trainer.callback_metrics.get("train_emotion_acc")
        if gender_acc is not None:
            self.train_accs_gender.append(gender_acc.detach().cpu().item())
        if emotion_acc is not None:
            self.train_accs_emotion.append(emotion_acc.detach().cpu().item())
    
def download_ravdess(song_dir="./ravdess_song", speech_dir="./ravdess_speech"):
    """Downloads and extracts the RAVDESS speech and song datasets if they are not already present."""
    if not os.path.exists(speech_dir):
        print('Downloading RAVDESS speech dataset...')
        subprocess.run([
            "curl", "-L", "-o", "./ravdess-emotional-speech-audio.zip",
            "https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-speech-audio"
        ], check=True)
        subprocess.run(
            ["unzip", "-o", "./ravdess-emotional-speech-audio.zip", "-d", speech_dir], 
            check=True
        )
    if not os.path.exists(song_dir):
        print('Downloading RAVDESS song dataset...')
        subprocess.run([
            "curl", "-L", "-o", "./ravdess-emotional-song-audio.zip",
            "https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-song-audio"
        ], check=True)
        subprocess.run(
            ["unzip", "-o", "./ravdess-emotional-song-audio.zip", "-d", song_dir], 
            check=True
        )
    
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./ravdess',
                 batch_size=32, train_split=0.8, 
                 num_workers=0, pin_memory=False,
                 embeddings_source='cached', cache_dir=None, 
                 vggish_model=None, device=None
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.dataset = self._index_data(data_dir) # list of file paths
        self.train_ds = None
        self.val_ds = None
        self.emb_src = embeddings_source
        
        self.cache_dir = cache_dir
        self.cache_files = []
        self.vggish_model = vggish_model
        self.max_examples = 4
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.emb_src not in {'raw', 'cached'}:
            raise ValueError("Embedding Source must be either 'raw' or 'cached'.")
    
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48000,
            new_freq=vggish_params.SAMPLE_RATE,
        )
        
        self.emotion_map = {
            0: 'neutral',
            1: 'calm',
            2: 'happy',
            3: 'sad',
            4: 'angry',
            5: 'fearful',
            6: 'disgust',
            7: 'surprised',
        }
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=vggish_params.SAMPLE_RATE,
            n_mels=vggish_params.NUM_MEL_BINS,
            n_fft=int(vggish_params.STFT_WINDOW_LENGTH_SECONDS * vggish_params.SAMPLE_RATE),
            hop_length=int(vggish_params.STFT_HOP_LENGTH_SECONDS * vggish_params.SAMPLE_RATE),
            f_min=vggish_params.MEL_MIN_HZ,
            f_max=vggish_params.MEL_MAX_HZ,
        )
        
        if self.emb_src == "cached":
            if not self.vggish_model:
                raise ValueError("vggish_model must be provided when embedding_source='cached'.")
            if not self.cache_dir:
                raise ValueError("cache_dir must be provided when embedding_source='cached'.")

            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "sample_*.pt")))

            expected = {os.path.join(self.cache_dir, f"sample_{i:05d}.pt") for i in range(len(self.dataset))}
            existing = set(self.cache_files)
            missing = sorted(expected - existing)

            if missing:
                missing_indices = [int(os.path.basename(p)[7:12]) for p in missing]
                self.build_embedding_cache(
                    self.vggish_model,
                    split="all",
                    overwrite=False,
                    dtype="float32",
                    indices=missing_indices,
                )

            self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "sample_*.pt")))
            if len(self.cache_files) != len(self.dataset):
                raise RuntimeError(f"Cache incomplete: {len(self.cache_files)} files for {len(self.dataset)} samples.")
    
    def _index_data(self, data_dir):
        """Recursively scans the given directory for .wav files and returns a sorted list of unique file paths. Handles duplicates by preferring top-level Actor_* files over nested ones."""
        all_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".wav"):
                    all_paths.append(os.path.join(root, file))

        # Prefer top-level Actor_* over nested audio_speech_actors_01-24/Actor_*
        def source_rank(path):
            return 1 if "/audio_speech_actors_01-24/" in path.replace("\\", "/") else 0

        dedup = {}
        for path in sorted(all_paths, key=source_rank):
            key = os.path.basename(path)  # RAVDESS filename is unique clip id
            if key not in dedup:
                dedup[key] = path

        file_paths = sorted(dedup.values())

        if not file_paths:
            raise ValueError(f"No .wav files found under {data_dir}.")

        print(f"Indexed {len(file_paths)} unique audio files from {data_dir} "
            f"(removed {len(all_paths) - len(file_paths)} duplicates).")
        return file_paths

    def _create_emotion_label(self, path):
        """Extracts the emotion label from the RAVDESS filename. Returns an integer in [0, 7] corresponding to the emotion category."""
        emotion_code = int(path.split("-")[2])
        return emotion_code - 1

    def _create_gender_label(self, path):
        """Extracts the gender label from the RAVDESS filename. Returns an integer: 0 or 1 corresponding to Male and Female respectively."""
        actor_code = int(path.split("-")[-1].split(".")[0])
        return 1 if actor_code % 2 == 0 else 0

    def _load_audio(self, file_path):
        """Loads audio from the given file path, resampling if necessary. Returns a tensor of shape [C, L] with values in [-1.0, 1.0]."""
        try:
            audio, fs = torchaudio.load(file_path)
        except Exception:
            audio, fs = librosa.load(file_path, sr=None, mono=False)
            audio = torch.tensor(audio, dtype=torch.float32)
            self.fs = fs
            
        if fs != vggish_params.SAMPLE_RATE:
            audio = self.resampler(audio)
            self.fs = vggish_params.SAMPLE_RATE
        # torchaudio.load returns float tensors already scaled to [-1, 1].
        # Avoid applying an extra 1/32768 factor.
        return audio
        # samples = audio / 32768.0
        # return samples 

    def _mel_spectrogram_examples(self, audio):
        """Converts raw audio to log-mel spectrogram examples suitable for VGGish input. Returns a tensor of shape [num_examples, 1, 96, 64]."""
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)

        if audio.ndim > 1:
            audio = torch.mean(audio, dim=0)
        audio = audio.to(torch.float32)

        mel_spec = self.mel_transform(audio)
        log_offset = getattr(vggish_params, 'LOG_OFFSET', 0.01)
        log_mel = torch.log(mel_spec + log_offset)
        log_mel = log_mel.transpose(0, 1).contiguous()

        features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
        window_len = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        hop_len = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
        required_frames = (self.max_examples - 1) * hop_len + window_len
        total_frames, _ = log_mel.shape

        if total_frames < required_frames:
            pad_amount = required_frames - total_frames
            log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, pad_amount), mode='constant', value=0.0)
        else:
            log_mel = log_mel[:required_frames, :]

        examples = []
        for i in range(self.max_examples):
            start = i * hop_len
            end = start + window_len
            examples.append(log_mel[start:end, :])

        stacked = torch.stack(examples, dim=0).to(torch.float32)
        return stacked.unsqueeze(1)
    
    def setup(self, stage=None):
        """Ensures the dataset is indexed and the train/val splits are created. This is called by PyTorch Lightning before training/validation starts."""
        if self.dataset is None:
            self.dataset = self.cache_files if self.emb_src == 'cached' else self._index_data(self.data_dir)

        if self.train_ds is None or self.val_ds is None:
            train_size = int(self.train_split * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_subset, val_subset = random_split(self.dataset, [train_size, val_size])
            self.train_ds = Subset(self, train_subset.indices)
            self.val_ds = Subset(self, val_subset.indices)

    def _get_split_indices(self, split):
        """Returns the list of dataset indices corresponding to the specified split ('train', 'val', or 'all'). This is used for building the embedding cache for a specific split."""
        self.setup()

        if split == 'all':
            return list(range(len(self.dataset)))
        if split == 'train':
            return list(self.train_ds.indices)
        if split == 'val':
            return list(self.val_ds.indices)
        raise ValueError("split must be one of: 'all', 'train', 'val'.")
        
    def build_embedding_cache(self, vggish_model, split='all', overwrite=False, dtype='float32', indices=None):
        """Precomputes and saves VGGish embeddings for the specified split ('train', 'val', or 'all') to disk. If overwrite=False, it will skip already cached samples. The dtype argument controls the precision of the saved embeddings (float32 or float16). If indices is provided, it should be a list of dataset indices to build the cache for (overriding the split argument)."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
        }
        if dtype not in dtype_map:
            raise ValueError("dtype must be 'float32' or 'float16'.")

        os.makedirs(self.cache_dir, exist_ok=True)
        vggish_model.eval()

        if indices is None:
            indices = self._get_split_indices(split)
        # print(indices)
        saved = 0
        skipped = 0
        # print(f"Building embedding cache for {len(indices)} samples in split '{split}' (overwrite={overwrite})...")
        with torch.no_grad():
            for idx in tqdm(indices, desc=f"Caching {len(self.dataset)} embeddings ({split})", unit="file"):
                # print(f"Processing file {idx}: {self.dataset[idx]}")
                save_path = os.path.join(self.cache_dir, f"sample_{idx:05d}.pt")
                if os.path.exists(save_path) and not overwrite:
                    skipped += 1
                    continue
                
                audio = self._load_audio(self.dataset[idx])
                mel_spec = self._mel_spectrogram_examples(audio)
                mel_spec = mel_spec.to(device=self.device, dtype=torch.float32)
                gender_label = self._create_gender_label(self.dataset[idx])
                emotion_label = self._create_emotion_label(self.dataset[idx])
                embedding = vggish_model(mel_spec)
                # Preserve temporal windows for downstream TCN receptive field.
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
                elif embedding.ndim > 2:
                    embedding = embedding.reshape(-1, embedding.shape[-1])

                embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

                torch.save(
                    {
                        'embedding': embedding.to(dtype_map[dtype]).cpu(),
                        'gender_label': gender_label.to(torch.long).cpu() if torch.is_tensor(gender_label) else torch.tensor(gender_label, dtype=torch.long),
                        'emotion_label': emotion_label.to(torch.long).cpu() if torch.is_tensor(emotion_label) else torch.tensor(emotion_label, dtype=torch.long),
                        'source_path': self.dataset[idx],
                        'source_index': idx
                    },
                    save_path,
                )
                del audio, mel_spec, embedding
                if idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                saved += 1
                # print(idx, self.dataset[idx], "->", save_path)
                # CRITICAL: Manual memory cleanup
            
        self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "sample_*.pt")))
        print(f"Saved {saved} embeddings to {self.cache_dir} (skipped {skipped}).")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.emb_src == 'cached':
            if idx >= len(self.cache_files):
                raise IndexError(
                    f"Cache index out of range: idx={idx}, cache_files={len(self.cache_files)}, dataset={len(self.dataset)}"
                )

            cache_path = self.cache_files[idx]
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Missing cache file for idx={idx}: {cache_path}")

            item = torch.load(cache_path, map_location='cpu')
            embedding = item['embedding'] if 'embedding' in item else item['embeddings']
            # Only squeeze singleton temporal dimension; keep real sequence length (e.g., 4).
            if embedding.ndim > 1 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            gender_label = item['gender_label']
            emotion_label = item['emotion_label']

            return embedding.to(torch.float32), gender_label, emotion_label

        audio = self._load_audio(self.dataset[idx])
        mel_spec = self._mel_spectrogram_examples(audio)
        return mel_spec, self._create_gender_label(self.dataset[idx]), self._create_emotion_label(self.dataset[idx])

    def train_dataloader(self):
        """Returns a DataLoader for the training split. This is called by PyTorch Lightning during training."""
        self.setup()
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Returns a DataLoader for the validation split. This is called by PyTorch Lightning during validation."""
        self.setup()
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

class SpeechClassifier(pl.LightningModule):
    """
    PyTorch Lightning module that defines the model architecture, training loop, validation loop, and optimization for the speech emotion and gender classification task. 
    It uses a pretrained VGGish model for feature extraction and a TCN-based temporal backbone for classification.
    """
    def __init__(self, 
         pretrained_vggish,
         tcn_model,
         learning_rate=1e-3,
         device=None,
         batch_size=32,
         backbone_out_channels=64,
         emotion_map=None,
         gender_loss_weight=1.0,
         emotion_loss_weight=1.0,
         activate_ablation=False
    ):
        super().__init__()
        
        # Check for GPU availability and set device
        if device is None:
            self.compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.compute_device = device
        
        
        if emotion_map is None:
            emotion_map = {
                0: 'neutral',
                1: 'calm',
                2: 'happy',           
                3: 'sad',
                4: 'angry',
                5: 'fearful',
                6: 'disgust',
                7: 'surprised',
            }
        
        self.ablation_mode = activate_ablation
        
        num_emotions = len(emotion_map)
        if tcn_model is None:
            raise ValueError("tcn_model must be provided (cannot be None).")

        # Setup Model Components
        self.vggish = pretrained_vggish  # Frozen feature extractor
        self.temporal_backbone = tcn_model
        self.gender_head = nn.Linear(backbone_out_channels, 1)
        self.emotion_head = nn.Linear(backbone_out_channels, num_emotions)
        self.batch_size = batch_size
        
        self.emotion_map = emotion_map
        
        # Setup Evaluation Metrics
        self.mc_conf_matrix = MulticlassConfusionMatrix(num_classes=num_emotions)  # 8 emotion
        self.bc_conf_matrix = ConfusionMatrix(num_classes=1, task="binary")  # Binary
    
        self.acc_emotion = MulticlassAccuracy(num_classes=num_emotions)
        self.acc_gender = BinaryAccuracy()
        
        self.f1_emotion = F1Score(task="multiclass", num_classes=num_emotions, average="macro")
        self.precision_emotion = Precision(num_classes=num_emotions, task="multiclass", average='macro')
        self.recall_emotion = Recall(num_classes=num_emotions, task="multiclass", average='macro')
        self.f1_gender = F1Score(task="binary")
        self.precision_gender = Precision(task="binary")
        self.recall_gender = Recall(task="binary")
        
        # 2. Hyperparameters
        self.lr = learning_rate
        self.gender_criterion = nn.BCEWithLogitsLoss()
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.gender_loss_weight = gender_loss_weight
        self.emotion_loss_weight = emotion_loss_weight
        self._has_val_updates = False
        
        # Freezing VGGish
        for param in self.vggish.parameters():
            param.requires_grad = False
            

    def _encode_vggish_embeddings(self, mel):
        """
        Encodes raw mel spectrograms into VGGish embeddings. 
        Handles both single examples and batches, as well as cases where temporal windows are present. 
        Expected input shapes: [B, T, F] for precomputed mel spectrograms or [B, S, 1, 96, 64] for raw spectrograms. 
        Returns embeddings of shape [B, T, E] or [B, E] depending on input.
        """
        if mel.ndim == 5:
            batch_size, num_windows = mel.shape[:2]
            mel = mel.reshape(batch_size * num_windows, *mel.shape[2:])
            embeddings = self.vggish(mel)
            return embeddings.reshape(batch_size, num_windows, -1)

        if mel.ndim == 4:
            embeddings = self.vggish(mel)
            return embeddings.unsqueeze(1)

        return self.vggish(mel)

    def forward(self, x):
        """
        Forward pass through the model. Handles both raw spectrograms and precomputed embeddings.
        Cached embeddings path. Expected shapes: [B, T, F] or [B, F].
        """    
        if x.ndim == 3:
            embeddings = x
        if x.ndim == 2:
            embeddings = x.unsqueeze(1)
        elif x.ndim == 5:
            # Raw spectrogram path. Expected shape: [B, S, 1, 96, 64].
            with torch.no_grad():
                B, S, C, H, W = x.shape
                x = x.view(B * S, C, H, W)
                emb = self.vggish(x)
                embeddings = emb.view(B, S, -1)

        # Conv1d expects [B, C, T]. Current embeddings are [B, T, F].
        if self.ablation_mode:
            tcn_in = torch.randn_like(embeddings).permute(0, 2, 1).contiguous()  # Random noise input for ablation
        else:
            tcn_in = embeddings.permute(0, 2, 1).contiguous()
        shared_features = self.temporal_backbone(tcn_in)
        pooled = shared_features.mean(dim=2)
        return self.gender_head(pooled), self.emotion_head(pooled)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step. Expects batch to be a tuple of (embeddings, gender_labels, emotion_labels).
        Computes losses, updates metrics, and logs results.
        """
        # Correctly unpack the 3 items from SpeechDataModule
        emb, gender, emotion = batch

        gender_logits, emotion_logits = self(emb)
        gender_logits = gender_logits.squeeze(-1)
        gender_targets = gender.float()

        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        emotion_loss = self.emotion_criterion(emotion_logits, emotion.long())
        loss = (
            self.gender_loss_weight * gender_loss
            + self.emotion_loss_weight * emotion_loss
        )

        gender_preds = (torch.sigmoid(gender_logits) > 0.5).long()
        gender_acc = (gender_preds == gender.long()).float().mean()
        emotion_preds = torch.argmax(emotion_logits, dim=1)
        emotion_acc = (emotion_preds == emotion.long()).float().mean()

        self.log_dict(
            {
                "train_loss": loss,
                "train_gender_loss": gender_loss,
                "train_emotion_loss": emotion_loss,
                "train_gender_acc": gender_acc,
                "train_emotion_acc": emotion_acc,
            },
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_start(self):
        """
        Resets the flag to track if any validation updates were made during this epoch. 
        This is used to determine whether to plot confusion matrices at the end of the epoch.
        """
        self._has_val_updates = False
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step. Expects batch to be a tuple of (embeddings, gender_labels, emotion_labels).
        Computes losses, updates metrics, and logs results. Also updates confusion matrices and F1/precision/recall metrics for both tasks. 
        """
        
        emb, gender, emotion = batch
        gender_logits, emotion_logits = self(emb)
        gender_logits = gender_logits.squeeze(-1)
        gender_targets = gender.float()
        gender_labels = gender.long()
        emotion_labels = emotion.long()
        gender_preds = (torch.sigmoid(gender_logits) > 0.5).long()
        emotion_preds = emotion_logits.argmax(dim=1)

        self._has_val_updates = True

        # Use the same discrete predictions for confusion matrices and metrics.
        self.bc_conf_matrix.update(gender_preds, gender_labels)
        self.mc_conf_matrix.update(emotion_preds, emotion_labels)
        self.f1_emotion.update(emotion_preds, emotion_labels)
        self.precision_emotion.update(emotion_preds, emotion_labels)
        self.recall_emotion.update(emotion_preds, emotion_labels)
        self.f1_gender.update(gender_preds, gender_labels)
        self.precision_gender.update(gender_preds, gender_labels)
        self.recall_gender.update(gender_preds, gender_labels)

        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
        loss = (
            self.gender_loss_weight * gender_loss
            + self.emotion_loss_weight * emotion_loss
        )

        emotion_acc = self.acc_emotion(emotion_preds, emotion_labels)
        gender_acc = self.acc_gender(gender_preds, gender_labels)

        self.log_dict(
            {
                "val_loss": loss,
                "val_gender_loss": gender_loss,
                "val_emotion_loss": emotion_loss,
                "val_gender_acc": gender_acc,
                "val_emotion_acc": emotion_acc,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        """Configures the optimizer for training. Only parameters of the temporal backbone and classification heads are optimized, while the VGGish feature extractor is frozen."""
        params = (
            list(self.temporal_backbone.parameters())
            + list(self.gender_head.parameters())
            + list(self.emotion_head.parameters())
        )
        return torch.optim.Adam(params, lr=self.lr)
    
    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, checks if this is the last epoch and if any validation updates were made. 
        If so, computes and plots the confusion matrices for both tasks, as well as prints out the F1 score, precision, and recall for emotion
        """
        
        should_plot = (
            self.current_epoch == self.trainer.max_epochs - 1
            and self._has_val_updates
            and not self.trainer.sanity_checking
        )
            
        if should_plot:
            bcm = self.bc_conf_matrix.compute()
            mccm = self.mc_conf_matrix.compute()
            f1_emotion = self.f1_emotion.compute()
            precision_emotion = self.precision_emotion.compute()
            recall_emotion = self.recall_emotion.compute()
            f1_gender = self.f1_gender.compute()
            precision_gender = self.precision_gender.compute()
            recall_gender = self.recall_gender.compute()
            
            fig_1, _ = self.bc_conf_matrix.plot()
            fig_2, _ = self.mc_conf_matrix.plot()
            fig_1.suptitle(f"Binary Confusion Matrix - Epoch {self.current_epoch}", fontsize=14, fontweight='bold')
            fig_2.suptitle(f"Multiclass Confusion Matrix - Epoch {self.current_epoch}", fontsize=14, fontweight='bold')
            
            os.mkdir("./confusion_metrics") if not os.path.exists("./confusion_metrics") else None
            fig_1.savefig(f"./confusion_metrics/epoch_{self.current_epoch}_binary.png")
            fig_2.savefig(f"./confusion_metrics/epoch_{self.current_epoch}_multi.png")
            
            print(f"Emotion Classification Metrics")
            print(f"=" * 30)
            print(f"    --> F1 Score: {f1_emotion:.4f}")
            print(f"    --> Precision: {precision_emotion:.4f}")
            print(f"    --> Recall: {recall_emotion:.4f}")
            
            print(f"Gender Classification Metrics")
            print(f"=" * 30)
            print(f"    --> F1 Score: {f1_gender:.4f}")
            print(f"    --> Precision: {precision_gender:.4f}")
            print(f"    --> Recall: {recall_gender:.4f}")

        # Always reset epoch-scoped metric state to prevent leakage into later epochs.
        self.bc_conf_matrix.reset()
        self.mc_conf_matrix.reset()
        self.acc_emotion.reset()
        self.acc_gender.reset()
        self.f1_emotion.reset()
        self.precision_emotion.reset()
        self.recall_emotion.reset()
        self.f1_gender.reset()
        self.precision_gender.reset()
        self.recall_gender.reset()
            
        return super().on_validation_epoch_end()
    
