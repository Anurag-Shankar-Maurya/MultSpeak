import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from scipy import signal
import torch
import torchaudio
import os
import urllib.request
import zipfile

from speechbrain.inference import EncoderClassifier, SpeakerRecognition


class VoiceProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_trained = False

        # Initialize SpeechBrain models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create cache directory if it doesn't exist
        os.makedirs("pretrained_models", exist_ok=True)

        # Download and load speaker embedding model
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

        # For verification (optional)
        self.verification_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb-verification",
            run_opts={"device": self.device}
        )

    def train_scaler(self, audio_samples):
        """Train the scaler on a diverse set of samples"""
        features_list = []
        for audio, sr in audio_samples:
            features = self.extract_features(audio, sr)
            features_list.append(features)

        if features_list:
            features_array = np.vstack(features_list)
            self.scaler.fit(features_array)
            self.scaler_trained = True
            return True
        return False

    def preprocess_audio(self, audio, sample_rate):
        """Remove noise and normalize the audio"""
        try:
            # Resample if necessary
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # High-pass filter to remove low-frequency noise
            b, a = self._butter_highpass(80, sample_rate, order=5)
            audio = self._apply_filter(b, a, audio)

            # Add envelope detection to remove silence
            audio_envelope = np.abs(audio)
            threshold = 0.005
            audio[audio_envelope < threshold] = 0

            # Apply noise reduction
            audio = self._reduce_noise(audio, sample_rate)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            return audio, sample_rate
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            return audio, sample_rate

    def _butter_highpass(self, cutoff, fs, order=5):
        """Design a high-pass Butterworth filter"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _apply_filter(self, b, a, audio):
        """Apply filter to the audio"""
        return signal.filtfilt(b, a, audio)

    def _reduce_noise(self, audio, sample_rate):
        """Simple noise reduction using spectral gating"""
        try:
            noise_len = min(int(0.5 * sample_rate), len(audio) // 4)
            if noise_len > 0:
                noise_profile = audio[:noise_len]
                noise_threshold = np.mean(np.abs(noise_profile)) * 2
                mask = np.abs(audio) > noise_threshold
                soft_mask = np.clip(np.abs(audio) / noise_threshold - 1, 0, 1)
                return audio * soft_mask
            return audio
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return audio

    def extract_features(self, audio, sample_rate):
        """Extract traditional acoustic features as backup"""
        audio, sample_rate = self.preprocess_audio(audio, sample_rate)

        # MFCCs with delta
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=24)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
        delta2_mfccs_mean = np.mean(delta2_mfccs, axis=1)

        # Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec)
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)

        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        centroid_mean = np.mean(spectral_centroid)

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        bandwidth_mean = np.mean(spectral_bandwidth)

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        rolloff_mean = np.mean(spectral_rolloff)

        # Concatenate all features
        features = np.concatenate([
            mfccs_mean, mfccs_std, delta_mfccs_mean, delta2_mfccs_mean,
            mel_mean, mel_std, spectral_contrast_mean,
            [zcr_mean], [zcr_std], [rms_mean], [rms_std],
            [centroid_mean], [bandwidth_mean], [rolloff_mean]
        ])

        return features

    def extract_embedding(self, audio, sample_rate):
        """Extract speaker embedding using SpeechBrain ECAPA-TDNN model"""
        try:
            # Preprocess audio
            audio, sample_rate = self.preprocess_audio(audio, sample_rate)

            # Convert to tensor
            waveform = torch.FloatTensor(audio).unsqueeze(0)

            # Extract embedding using SpeechBrain
            with torch.no_grad():
                embeddings = self.embedding_model.encode_batch(waveform)
                embedding = embeddings[0].squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            print(f"Error extracting SpeechBrain embedding: {str(e)}")
            # Fallback to traditional features
            features = self.extract_features(audio, sample_rate)
            features = features.reshape(1, -1)

            if self.scaler_trained:
                embedding = self.scaler.transform(features).flatten()
            else:
                print("Warning: Scaler not trained. Consider calling train_scaler() first.")
                self.scaler.fit(features)
                embedding = self.scaler.transform(features).flatten()

            return embedding

    def verify_speakers(self, audio1, sr1, audio2, sr2):
        """Verify if two audio segments belong to the same speaker"""
        try:
            # Preprocess audio
            audio1, sr1 = self.preprocess_audio(audio1, sr1)
            audio2, sr2 = self.preprocess_audio(audio2, sr2)

            # Convert to tensors
            waveform1 = torch.FloatTensor(audio1).unsqueeze(0)
            waveform2 = torch.FloatTensor(audio2).unsqueeze(0)

            # Use verification model
            score, prediction = self.verification_model.verify_batch(
                waveform1, waveform2
            )

            return score.item(), prediction.item()

        except Exception as e:
            print(f"Error in speaker verification: {str(e)}")
            # Fallback to cosine similarity of embeddings
            emb1 = self.extract_embedding(audio1, sr1)
            emb2 = self.extract_embedding(audio2, sr2)

            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

            return sim, sim > 0.5

    def augment_audio(self, audio, sample_rate):
        """Create augmented versions of the audio"""
        augmented = []

        try:
            # Pitch shift up
            audio_pitch_up = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=1)
            augmented.append((audio_pitch_up, sample_rate))

            # Pitch shift down
            audio_pitch_down = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-1)
            augmented.append((audio_pitch_down, sample_rate))

            # Time stretch
            audio_stretch = librosa.effects.time_stretch(audio, rate=0.9)
            augmented.append((audio_stretch, sample_rate))

            # Add small amount of noise
            noise = np.random.normal(0, 0.005, len(audio))
            audio_noise = audio + noise
            augmented.append((audio_noise, sample_rate))

        except Exception as e:
            print(f"Error in audio augmentation: {str(e)}")

        return augmented
