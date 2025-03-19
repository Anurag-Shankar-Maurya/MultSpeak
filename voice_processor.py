import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from scipy import signal


class VoiceProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_trained = False

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
            b, a = self._butter_highpass(80, sample_rate, order=5)  # Increased from 50
            audio = self._apply_filter(b, a, audio)

            # Add envelope detection to remove silence
            audio_envelope = np.abs(audio)
            threshold = 0.005  # Adjust based on your needs
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
            # Calculate noise profile from first 0.5 seconds or minimum audio length
            noise_len = min(int(0.5 * sample_rate), len(audio) // 4)
            if noise_len > 0:
                noise_profile = audio[:noise_len]

                # Get noise threshold
                noise_threshold = np.mean(np.abs(noise_profile)) * 2

                # Create a mask
                mask = np.abs(audio) > noise_threshold

                # Apply soft mask
                soft_mask = np.clip(np.abs(audio) / noise_threshold - 1, 0, 1)
                return audio * soft_mask
            return audio
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return audio

    def extract_features(self, audio, sample_rate):
        """Extract robust voice features"""
        audio, sample_rate = self.preprocess_audio(audio, sample_rate)

        # 1. Mel-frequency cepstral coefficients (MFCCs) with delta
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=24)  # Increased from 20
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)  # Added std features
        delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
        delta2_mfccs_mean = np.mean(delta2_mfccs, axis=1)

        # 2. Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec)
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)

        # 3. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        # 4. Tonnetz (tonal centroid features) - removed as it might add noise
        # tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
        # tonnetz_mean = np.mean(tonnetz, axis=1)

        # 5. Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)  # Added std feature

        # 6. Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)  # Added std feature

        # 7. Spectral centroid (brightness of sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        centroid_mean = np.mean(spectral_centroid)

        # 8. Spectral bandwidth (spread of spectrum around centroid)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        bandwidth_mean = np.mean(spectral_bandwidth)

        # 9. Spectral rolloff
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
        """Extract improved voice embedding"""
        features = self.extract_features(audio, sample_rate)
        features = features.reshape(1, -1)

        # Check if scaler is already fitted
        if self.scaler_trained:
            # Use the pre-trained scaler
            embedding = self.scaler.transform(features).flatten()
        else:
            # Fallback - fit on current sample (not ideal, but prevents errors)
            print("Warning: Scaler not trained. Consider calling train_scaler() first.")
            self.scaler.fit(features)
            embedding = self.scaler.transform(features).flatten()

        return embedding

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