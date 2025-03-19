import os
import numpy as np
import sounddevice as sd
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use("TkAgg")

from voice_processor import VoiceProcessor
from speaker_database import SpeakerDatabase
from speaker_diarization import SpeakerDiarization


class SpeakerRecognitionSystem:
    def __init__(self):
        # Initialize components
        self.voice_processor = VoiceProcessor()
        self.speaker_db = SpeakerDatabase()
        self.diarization = SpeakerDiarization(self.voice_processor, self.speaker_db)

        # Default settings
        self.sample_rate = 16000
        self.db_file = "speaker_database.pkl"
        self.recording = False
        self.recorded_audio = None
        self.current_speaker = None

        # Load database if exists
        self.load_database()

        # Train scaler if we have data
        if self.speaker_db.raw_audio:
            samples = self.speaker_db.get_all_raw_audio_samples()
            if samples:
                self.voice_processor.train_scaler(samples)

    def load_database(self):
        """Load speaker database from file"""
        if os.path.exists(self.db_file):
            success = self.speaker_db.load_database(self.db_file)
            return success
        return False

    def save_database(self):
        """Save speaker database to file"""
        self.speaker_db.save_database(self.db_file)

    def start_recording(self, duration=5):
        """Start recording audio"""
        self.recording = True
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * self.sample_rate),
                       samplerate=self.sample_rate,
                       channels=1,
                       dtype='float32')
        sd.wait()
        self.recording = False
        self.recorded_audio = audio.flatten()
        print("Recording complete!")
        return self.recorded_audio

    def load_audio_file(self, file_path):
        """Load audio from file"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            self.recorded_audio = audio
            return audio, self.sample_rate
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            return None, None

    def process_audio(self, audio=None, sample_rate=None):
        """Process audio and extract embedding"""
        if audio is None:
            audio = self.recorded_audio
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Preprocess audio
        audio, sample_rate = self.voice_processor.preprocess_audio(audio, sample_rate)

        # Extract embedding
        embedding = self.voice_processor.extract_embedding(audio, sample_rate)

        return audio, sample_rate, embedding

    def add_speaker(self, speaker_name, audio=None, sample_rate=None):
        """Add a new speaker to the database"""
        if audio is None:
            audio = self.recorded_audio
        if sample_rate is None:
            sample_rate = self.sample_rate

        if audio is None:
            print("No audio recorded or loaded")
            return False

        # Process audio and extract embedding
        audio, sample_rate, embedding = self.process_audio(audio, sample_rate)

        # Add to database
        self.speaker_db.add_speaker(speaker_name, embedding, raw_audio=audio, sample_rate=sample_rate)

        # Save database
        self.save_database()

        # Train scaler if this is the first speaker
        if len(self.speaker_db.speakers) == 1:
            samples = self.speaker_db.get_all_raw_audio_samples()
            self.voice_processor.train_scaler(samples)

        return True

    def identify_speaker(self, audio=None, sample_rate=None):
        """Identify the speaker in the audio"""
        if audio is None:
            audio = self.recorded_audio
        if sample_rate is None:
            sample_rate = self.sample_rate

        if audio is None:
            print("No audio recorded or loaded")
            return None, 0

        # Process audio and extract embedding
        audio, sample_rate, embedding = self.process_audio(audio, sample_rate)

        # Find closest match
        speaker_name, confidence = self.speaker_db.find_closest_match(embedding)

        return speaker_name, confidence

    def perform_diarization(self, audio=None, sample_rate=None):
        """Perform speaker diarization on audio"""
        if audio is None:
            audio = self.recorded_audio
        if sample_rate is None:
            sample_rate = self.sample_rate

        if audio is None:
            print("No audio recorded or loaded")
            return []

        # Run diarization
        results = self.diarization.diarize(audio, sample_rate)

        return results

    def generate_diarization_plot(self, results, figure=None):
        """Generate a plot of diarization results"""
        if not results:
            return None

        if figure is None:
            figure = plt.figure(figsize=(10, 4))
        else:
            figure.clear()

        ax = figure.add_subplot(111)

        # Get unique speakers
        speakers = list(set([r[2] for r in results]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(speakers)))
        speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}

        # Plot segments
        for start, end, speaker, conf in results:
            ax.barh(y=speaker, width=end - start, left=start,
                    color=speaker_colors[speaker], alpha=0.7)

        ax.set_title("Speaker Diarization")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Speaker")
        ax.grid(True, linestyle='--', alpha=0.7)

        return figure

    def remove_speaker(self, speaker_name):
        """Remove a speaker from the database"""
        success = self.speaker_db.remove_speaker(speaker_name)
        if success:
            self.save_database()
        return success

    def get_speaker_list(self):
        """Get list of all speakers in the database"""
        return list(self.speaker_db.speakers.keys())

    def get_speaker_samples_count(self, speaker_name):
        """Get number of samples for a speaker"""
        return self.speaker_db.get_speaker_samples_count(speaker_name)


def main():
    """Main function to run the application"""
    import gui
    app = gui.SpeakerRecognitionGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
