import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler


class VoiceProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_audio(self, audio, sample_rate):
        """Remove noise and normalize the audio"""
        try:
            # Resample if necessary
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # Basic noise reduction through high-pass filter
            b, a = self._butter_highpass(50, sample_rate, order=5)
            audio = self._apply_filter(b, a, audio)

            # Normalize the audio
            audio = librosa.util.normalize(audio)

            return audio, sample_rate
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            return audio, sample_rate

    def _butter_highpass(self, cutoff, fs, order=5):
        """Design a high-pass filter using Butterworth filter"""
        from scipy import signal
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _apply_filter(self, b, a, audio):
        """Apply filter to the audio"""
        from scipy import signal
        return signal.filtfilt(b, a, audio)

    def extract_features(self, audio, sample_rate):
        """Extract voice features from the audio"""
        # Preprocess the audio
        audio, sample_rate = self.preprocess_audio(audio, sample_rate)

        # Extract features
        # 1. Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # 2. Spectral centroid (related to brightness of the sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_centroid_mean = np.mean(spectral_centroid)

        # 3. Spectral rolloff (measure of the shape of the signal)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_rolloff_mean = np.mean(spectral_rolloff)

        # 4. Zero crossing rate (related to the pitch)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zero_crossing_rate)

        # 5. Chromagram (pitch content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)

        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            [spectral_centroid_mean], [spectral_rolloff_mean], [zcr_mean],
            chroma_mean
        ])

        return features

    def extract_embedding(self, audio, sample_rate):
        """Extract voice embedding (digital fingerprint) from features"""
        # Extract features
        features = self.extract_features(audio, sample_rate)

        # In a real application, we would use a pre-trained deep learning model
        # to extract embeddings. For simplicity, we'll use our extracted features
        # as the embedding after normalization.
        features = features.reshape(1, -1)

        # Normalize features
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            self.scaler.fit(features)

        embedding = self.scaler.transform(features).flatten()

        return embedding
