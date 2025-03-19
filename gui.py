import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use("TkAgg")
import sounddevice as sd
import librosa
import wave
import soundfile as sf

from main import SpeakerRecognitionSystem


class SpeakerRecognitionGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize system
        self.system = SpeakerRecognitionSystem()

        # Setup GUI
        self.title("Speaker Recognition System")
        self.geometry("800x600")
        self.configure(bg="#f0f0f0")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create frames for each tab
        self.speaker_frame = ttk.Frame(self.notebook)
        self.identify_frame = ttk.Frame(self.notebook)
        self.diarize_frame = ttk.Frame(self.notebook)

        # Add frames to notebook
        self.notebook.add(self.speaker_frame, text="Speaker Management")
        self.notebook.add(self.identify_frame, text="Speaker Identification")
        self.notebook.add(self.diarize_frame, text="Speaker Diarization")

        # Setup each tab
        self.setup_speaker_tab()
        self.setup_identify_tab()
        self.setup_diarize_tab()

        # Variables
        self.recording = False
        self.recording_thread = None
        self.audio_file = None
        self.diarization_results = None

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Refresh speaker list
        self.refresh_speaker_list()

    def setup_speaker_tab(self):
        """Setup the speaker management tab"""
        # Left frame for controls
        left_frame = ttk.Frame(self.speaker_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Speaker list
        ttk.Label(left_frame, text="Registered Speakers:").pack(anchor=tk.W, pady=(0, 5))

        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.speaker_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE)
        self.speaker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.speaker_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.speaker_listbox.config(yscrollcommand=scrollbar.set)

        # Speaker details
        details_frame = ttk.LabelFrame(left_frame, text="Speaker Details")
        details_frame.pack(fill=tk.X, pady=10)

        ttk.Label(details_frame, text="Samples:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.samples_label = ttk.Label(details_frame, text="0")
        self.samples_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Remove Speaker", command=self.remove_speaker).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh List", command=self.refresh_speaker_list).pack(side=tk.LEFT, padx=5)

        # Right frame for adding new speakers
        right_frame = ttk.LabelFrame(self.speaker_frame, text="Add New Speaker")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(right_frame, text="Speaker Name:").pack(anchor=tk.W, pady=(10, 5))
        self.speaker_name_var = tk.StringVar()
        ttk.Entry(right_frame, textvariable=self.speaker_name_var).pack(fill=tk.X, padx=5)

        ttk.Label(right_frame, text="Audio Source:").pack(anchor=tk.W, pady=(10, 5))

        # Audio source buttons
        audio_frame = ttk.Frame(right_frame)
        audio_frame.pack(fill=tk.X, pady=5)

        ttk.Button(audio_frame, text="Record (5s)", command=self.record_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(audio_frame, text="Load File", command=self.load_audio_file).pack(side=tk.LEFT, padx=5)

        self.audio_status_var = tk.StringVar()
        self.audio_status_var.set("No audio recorded or loaded")
        ttk.Label(right_frame, textvariable=self.audio_status_var).pack(anchor=tk.W, pady=5)

        ttk.Button(right_frame, text="Add Speaker", command=self.add_speaker).pack(anchor=tk.W, pady=10)

        # Bind listbox selection
        self.speaker_listbox.bind('<<ListboxSelect>>', self.on_speaker_select)

    def setup_identify_tab(self):
        """Setup the speaker identification tab"""
        # Top frame for controls
        top_frame = ttk.Frame(self.identify_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Audio Source:").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Record (5s)", command=self.record_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Load File", command=self.load_audio_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Identify Speaker", command=self.identify_speaker).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.identify_frame, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(results_frame, text="Identified Speaker:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.identified_speaker_var = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.identified_speaker_var).grid(row=0, column=1, sticky=tk.W, padx=5,
                                                                                pady=5)

        ttk.Label(results_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.confidence_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Audio status
        self.identify_audio_status_var = tk.StringVar()
        self.identify_audio_status_var.set("No audio recorded or loaded")
        ttk.Label(self.identify_frame, textvariable=self.identify_audio_status_var).pack(anchor=tk.W, padx=10, pady=5)

    def setup_diarize_tab(self):
        """Setup the speaker diarization tab"""
        # Top frame for controls
        top_frame = ttk.Frame(self.diarize_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Audio Source:").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Record (10s)", command=lambda: self.record_audio(10)).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Load File", command=self.load_audio_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Perform Diarization", command=self.perform_diarization).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.diarize_frame, text="Diarization Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create figure for plotting
        self.diarize_figure = plt.figure(figsize=(8, 4))
        self.diarize_canvas = FigureCanvasTkAgg(self.diarize_figure, master=results_frame)
        self.diarize_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Audio status
        self.diarize_audio_status_var = tk.StringVar()
        self.diarize_audio_status_var.set("No audio recorded or loaded")
        ttk.Label(self.diarize_frame, textvariable=self.diarize_audio_status_var).pack(anchor=tk.W, padx=10, pady=5)

    def refresh_speaker_list(self):
        """Refresh the speaker list"""
        self.speaker_listbox.delete(0, tk.END)
        speakers = self.system.get_speaker_list()
        for speaker in speakers:
            self.speaker_listbox.insert(tk.END, speaker)

    def on_speaker_select(self, event):
        """Handle speaker selection"""
        if self.speaker_listbox.curselection():
            index = self.speaker_listbox.curselection()[0]
            speaker = self.speaker_listbox.get(index)
            samples = self.system.get_speaker_samples_count(speaker)
            self.samples_label.config(text=str(samples))

    def record_audio(self, duration=5):
        """Record audio"""
        if self.recording:
            messagebox.showwarning("Recording", "Already recording!")
            return

        def record_thread():
            self.status_var.set(f"Recording for {duration} seconds...")
            self.recording = True
            self.system.start_recording(duration)
            self.recording = False
            self.status_var.set("Recording complete!")
            self.audio_status_var.set(f"Audio recorded ({duration}s)")
            self.identify_audio_status_var.set(f"Audio recorded ({duration}s)")
            self.diarize_audio_status_var.set(f"Audio recorded ({duration}s)")

        self.recording_thread = threading.Thread(target=record_thread)
        self.recording_thread.start()

    def load_audio_file(self):
        """Load audio from file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac")]
        )

        if file_path:
            self.status_var.set("Loading audio file...")
            audio, sr = self.system.load_audio_file(file_path)

            if audio is not None:
                self.audio_file = file_path
                filename = os.path.basename(file_path)
                self.audio_status_var.set(f"Loaded: {filename}")
                self.identify_audio_status_var.set(f"Loaded: {filename}")
                self.diarize_audio_status_var.set(f"Loaded: {filename}")
                self.status_var.set("Audio file loaded successfully")
            else:
                self.status_var.set("Failed to load audio file")

    def add_speaker(self):
        """Add a new speaker"""
        speaker_name = self.speaker_name_var.get().strip()

        if not speaker_name:
            messagebox.showerror("Error", "Please enter a speaker name")
            return

        if speaker_name in self.system.get_speaker_list():
            messagebox.showerror("Error", f"Speaker '{speaker_name}' already exists")
            return

        if self.system.recorded_audio is None:
            messagebox.showerror("Error", "No audio recorded or loaded")
            return

        self.status_var.set(f"Adding speaker '{speaker_name}'...")
        success = self.system.add_speaker(speaker_name)

        if success:
            self.status_var.set(f"Speaker '{speaker_name}' added successfully")
            self.refresh_speaker_list()
            self.speaker_name_var.set("")
        else:
            self.status_var.set(f"Failed to add speaker '{speaker_name}'")

    def remove_speaker(self):
        """Remove a speaker"""
        if not self.speaker_listbox.curselection():
            messagebox.showerror("Error", "No speaker selected")
            return

        index = self.speaker_listbox.curselection()[0]
        speaker_name = self.speaker_listbox.get(index)

        if messagebox.askyesno("Confirm", f"Are you sure you want to remove '{speaker_name}'?"):
            self.status_var.set(f"Removing speaker '{speaker_name}'...")
            success = self.system.remove_speaker(speaker_name)

            if success:
                self.status_var.set(f"Speaker '{speaker_name}' removed successfully")
                self.refresh_speaker_list()
            else:
                self.status_var.set(f"Failed to remove speaker '{speaker_name}'")

    def identify_speaker(self):
        """Identify the speaker"""
        if self.system.recorded_audio is None:
            messagebox.showerror("Error", "No audio recorded or loaded")
            return

        self.status_var.set("Identifying speaker...")
        speaker, confidence = self.system.identify_speaker()

        self.identified_speaker_var.set(speaker)
        self.confidence_var.set(f"{confidence:.2f}")

        self.status_var.set("Speaker identification complete")

    def perform_diarization(self):
        """Perform speaker diarization"""
        if self.system.recorded_audio is None:
            messagebox.showerror("Error", "No audio recorded or loaded")
            return

        self.status_var.set("Performing diarization...")

        # Run diarization in a thread to avoid UI freezing
        def diarize_thread():
            results = self.system.perform_diarization()
            self.diarization_results = results

            # Update plot
            self.system.generate_diarization_plot(results, self.diarize_figure)
            self.diarize_canvas.draw()

            self.status_var.set("Diarization complete")

        threading.Thread(target=diarize_thread).start()

    def on_close(self):
        """Handle window close event"""
        # Save database before closing
        self.system.save_database()
        self.destroy()


if __name__ == "__main__":
    app = SpeakerRecognitionGUI()
    app.mainloop()
