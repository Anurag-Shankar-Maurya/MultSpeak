import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import sounddevice as sd
import threading
import time
import speech_recognition as sr
import queue
from voice_processor import VoiceProcessor
from speaker_database import SpeakerDatabase
from gui import GUI


class MultSpeak:
    def __init__(self):
        self.voice_processor = VoiceProcessor()
        self.speaker_db = SpeakerDatabase()

        # Load existing database if available
        if os.path.exists('speaker_database.pkl'):
            self.speaker_db.load_database('speaker_database.pkl')

        # Initialize GUI
        self.root = tk.Tk()
        self.gui = GUI(self.root, self)

        # For real-time processing
        self.listening = False
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()

        # Update the GUI with existing speakers
        self.update_user_display()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle when the window is closed"""
        self.stop_listening()
        self.root.destroy()

    def record_sample(self, duration=10):
        """Record a voice sample for the specified duration"""
        self.gui.update_status("Recording voice sample... Please speak now")

        # Setup recording parameters
        sample_rate = 16000
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)

        # Show recording progress
        for i in range(duration):
            self.gui.update_status(f"Recording... {duration - i} seconds remaining")
            time.sleep(1)

        sd.wait()
        self.gui.update_status("Recording complete!")

        return recording.flatten(), sample_rate

    def process_audio_file(self, file_path):
        """Process an uploaded audio file"""
        try:
            self.gui.update_status(f"Processing audio file: {os.path.basename(file_path)}")
            audio, sample_rate = librosa.load(file_path, sr=None)
            return audio, sample_rate
        except Exception as e:
            self.gui.show_error(f"Error processing audio file: {str(e)}")
            return None, None

    def add_user_sample(self):
        """Add a user voice sample"""

        def process():
            audio, sample_rate = self.record_sample()
            if audio is not None:
                self.gui.prompt_for_user_name(audio, sample_rate)

        threading.Thread(target=process).start()

    def add_user_from_file(self):
        """Add a user voice sample from file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")]
        )

        if file_path:
            def process():
                audio, sample_rate = self.process_audio_file(file_path)
                if audio is not None:
                    self.gui.prompt_for_user_name(audio, sample_rate)

            threading.Thread(target=process).start()

    def register_user(self, audio, sample_rate, user_name, augment=True):
        """Register a new user to the database"""
        try:
            self.gui.update_status(f"Registering user: {user_name}")

            # Extract embedding from the original audio
            embedding = self.voice_processor.extract_embedding(audio, sample_rate)

            # Add the speaker with the embedding and raw audio
            self.speaker_db.add_speaker(user_name, embedding, audio, sample_rate)

            # Create augmented samples if requested
            if augment:
                self.gui.update_status("Creating augmented samples...")
                augmented_samples = self.voice_processor.augment_audio(audio, sample_rate)

                # Add each augmented sample
                for aug_audio, aug_sr in augmented_samples:
                    aug_embedding = self.voice_processor.extract_embedding(aug_audio, aug_sr)
                    self.speaker_db.add_speaker(user_name, aug_embedding, aug_audio, aug_sr)

                self.gui.update_status(f"Added {len(augmented_samples)} augmented samples")

            # Save the updated database
            self.speaker_db.save_database('speaker_database.pkl')

            # Update the GUI
            self.update_user_display()

            self.gui.update_status(f"User {user_name} registered successfully!")
            return True
        except Exception as e:
            self.gui.show_error(f"Error registering user: {str(e)}")
            return False

    def delete_selected_user(self):
        """Delete the selected user from the database"""
        # Get the selected item from the treeview
        selected_items = self.gui.users_list.selection()
        if not selected_items:
            self.gui.show_error("No user selected!")
            return

        # Get the username from the selected item
        user_name = self.gui.users_list.item(selected_items[0])['values'][0]

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete user '{user_name}' and all associated voice samples?"
        )

        if confirm:
            # Delete the user from the database
            if self.speaker_db.remove_speaker(user_name):
                self.gui.update_status(f"User {user_name} deleted successfully!")
                # Save the updated database
                self.speaker_db.save_database('speaker_database.pkl')
                # Update the GUI
                self.update_user_display()
            else:
                self.gui.show_error(f"Failed to delete user: {user_name}")

    def train_voice_processor(self):
        """Train the voice processor scaler using all available samples"""
        try:
            # Get all raw audio samples from the database
            all_samples = self.speaker_db.get_all_raw_audio_samples()

            if not all_samples:
                self.gui.show_error("No audio samples available! Please add user samples first.")
                return

            self.gui.update_status("Training voice processor...")

            # Start the progress bar
            self.gui.progress_bar.start()

            # Train the scaler in a separate thread to avoid blocking the GUI
            def train_thread():
                try:
                    success = self.voice_processor.train_scaler(all_samples)

                    if success:
                        self.gui.update_status("Voice processor trained successfully!")
                    else:
                        self.gui.show_error("Failed to train voice processor!")

                    # Stop the progress bar
                    self.gui.progress_bar.stop()
                except Exception as e:
                    self.gui.show_error(f"Error training voice processor: {str(e)}")
                    self.gui.progress_bar.stop()

            threading.Thread(target=train_thread).start()

        except Exception as e:
            self.gui.show_error(f"Error training voice processor: {str(e)}")
            self.gui.progress_bar.stop()

    def update_user_display(self):
        """Update the user list display with sample counts"""
        users = list(self.speaker_db.speakers.keys())
        sample_counts = {user: self.speaker_db.get_speaker_samples_count(user) for user in users}
        self.gui.update_user_list(users, sample_counts)

    def start_real_time_recognition(self):
        """Start real-time speech recognition and speaker identification"""
        if self.listening:
            self.gui.show_error("Already listening!")
            return

        if not self.speaker_db.speakers:
            self.gui.show_error("No users registered! Please add user samples first.")
            return

        self.listening = True
        self.gui.update_listening_status(True)

        # Start the listening thread
        threading.Thread(target=self.listen_continuously).start()

        # Start the processing thread
        threading.Thread(target=self.process_audio_queue).start()

    def stop_listening(self):
        """Stop real-time listening"""
        self.listening = False
        self.gui.update_listening_status(False)
        self.gui.update_status("Listening stopped")

    def listen_continuously(self):
        """Continuously listen for speech and add to queue"""
        # Initialize microphone
        mic = sr.Microphone()

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.gui.update_status("Listening for speech...")

            while self.listening:
                try:
                    # Listen for speech with a timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.gui.update_status(f"Error listening: {str(e)}")
                    time.sleep(1)

    def process_audio_queue(self):
        """Process audio from the queue"""
        while self.listening:
            try:
                # Get audio from queue with timeout
                audio = self.audio_queue.get(timeout=1)

                # Process the audio
                self.process_speech(audio)

            except queue.Empty:
                continue
            except Exception as e:
                self.gui.update_status(f"Error processing: {str(e)}")

    def process_speech(self, audio):
        """Process speech audio for transcription and speaker identification"""
        try:
            # Convert audio to numpy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to -1.0 to 1.0

            # Get speech-to-text
            text = self.recognizer.recognize_google(audio)

            # Extract embedding
            embedding = self.voice_processor.extract_embedding(audio_data, audio.sample_rate)

            # Get threshold value from GUI
            threshold_value = self.gui.threshold_var.get() / 100.0

            # Identify speaker
            speaker_name, confidence = self.speaker_db.find_closest_match(
                embedding,
                threshold=threshold_value
            )

            # Format confidence as percentage
            confidence_pct = int(confidence * 100)

            # Update GUI with results
            self.gui.add_speech_message(speaker_name, text, confidence_pct)

        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except sr.RequestError as e:
            self.gui.update_status(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            self.gui.update_status(f"Error processing speech: {str(e)}")


if __name__ == "__main__":
    app = MultSpeak()
    app.run()
