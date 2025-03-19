import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading


class GUI:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller

        # Set up the GUI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        self.root.title("MultSpeak - Real-time Speaker Recognition")
        self.root.geometry("800x600")

        # Create main frames
        self.setup_frame = ttk.LabelFrame(self.root, text="Setup - Add User Voice Samples", padding="10")
        self.setup_frame.pack(fill=tk.X, padx=10, pady=5)

        self.users_frame = ttk.LabelFrame(self.root, text="Registered Users", padding="10")
        self.users_frame.pack(fill=tk.X, padx=10, pady=5)

        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.chat_frame = ttk.LabelFrame(self.root, text="Real-time Conversation", padding="10")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.status_frame = ttk.Frame(self.root, padding="5")
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)

        # Setup section
        self.setup_ui_setup_section()

        # Users section
        self.setup_ui_users_section()

        # Control section
        self.setup_ui_control_section()

        # Chat section
        self.setup_ui_chat_section()

        # Status section
        self.setup_ui_status_section()

    def setup_ui_setup_section(self):
        """Set up the setup section"""
        # Buttons for adding users
        ttk.Button(
            self.setup_frame,
            text="Record Voice Sample",
            command=self.controller.add_user_sample
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            self.setup_frame,
            text="Upload Voice Sample",
            command=self.controller.add_user_from_file
        ).pack(side=tk.LEFT, padx=5)

        # Add a button for training the scaler
        ttk.Button(
            self.setup_frame,
            text="Train System",
            command=self.controller.train_voice_processor
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(
            self.setup_frame,
            text="Add multiple voice samples for each user (5+ recommended)",
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=20)

    def setup_ui_users_section(self):
        """Set up the users section"""
        self.users_list = ttk.Treeview(
            self.users_frame,
            columns=("User", "Samples"),
            show="headings",
            height=3
        )
        self.users_list.heading("User", text="Registered Users")
        self.users_list.heading("Samples", text="Samples Count")
        self.users_list.column("User", width=150)
        self.users_list.column("Samples", width=100)
        self.users_list.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.users_frame, orient=tk.VERTICAL, command=self.users_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.users_list.configure(yscrollcommand=scrollbar.set)

        # Add buttons for managing users
        button_frame = ttk.Frame(self.users_frame)
        button_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            button_frame,
            text="Delete User",
            command=self.controller.delete_selected_user
        ).pack(pady=5)

    def setup_ui_control_section(self):
        """Set up the control section"""
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Listening",
            command=self.controller.start_real_time_recognition
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            self.control_frame,
            text="Stop Listening",
            command=self.controller.stop_listening,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.listening_label = ttk.Label(
            self.control_frame,
            text="Not listening",
            foreground="red",
            font=("Arial", 10, "bold")
        )
        self.listening_label.pack(side=tk.LEFT, padx=20)

        # Add confidence threshold slider
        threshold_frame = ttk.Frame(self.control_frame)
        threshold_frame.pack(side=tk.RIGHT, padx=20)

        ttk.Label(
            threshold_frame,
            text="Confidence Threshold:"
        ).pack(side=tk.LEFT)

        self.threshold_var = tk.IntVar(value=30)
        self.threshold_slider = ttk.Scale(
            threshold_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.threshold_var,
            length=100
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        self.threshold_label = ttk.Label(
            threshold_frame,
            text="30%"
        )
        self.threshold_label.pack(side=tk.LEFT)

        # Update threshold label when slider changes
        self.threshold_var.trace_add("write", self.update_threshold_label)

    def update_threshold_label(self, *args):
        """Update the threshold label when slider changes"""
        value = self.threshold_var.get()
        self.threshold_label.config(text=f"{value}%")

    def setup_ui_chat_section(self):
        """Set up the chat section"""
        # Chat text area
        self.chat_text = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Arial", 10)
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.config(state=tk.DISABLED)

        # Set up tags for different speakers
        self.chat_text.tag_config("system", foreground="gray")
        self.chat_text.tag_config("header", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("speech", foreground="black", font=("Arial", 10, "italic"))
        self.chat_text.tag_config("confidence_high", foreground="green")
        self.chat_text.tag_config("confidence_medium", foreground="orange")
        self.chat_text.tag_config("confidence_low", foreground="red")

        # Add clear button
        ttk.Button(
            self.chat_frame,
            text="Clear Chat",
            command=self.clear_chat
        ).pack(pady=5)

    def clear_chat(self):
        """Clear the chat area"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def setup_ui_status_section(self):
        """Set up the status section"""
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            font=("Arial", 10),
            wraplength=780
        )
        self.status_label.pack(fill=tk.X)

        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            mode='indeterminate',
            length=780
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

    def update_status(self, message):
        """Update the status message"""

        def update():
            self.status_label.config(text=message)
            self.root.update()

        # Schedule the update on the main thread
        self.root.after(0, update)

    def show_error(self, message):
        """Show an error message"""
        messagebox.showerror("Error", message)

    def prompt_for_user_name(self, audio, sample_rate):
        """Prompt for a user name for the recorded sample"""

        def register():
            name = name_entry.get().strip()
            if name:
                dialog.destroy()
                self.controller.register_user(audio, sample_rate, name)
            else:
                messagebox.showwarning("Warning", "Please enter a name for the user")

        dialog = tk.Toplevel(self.root)
        dialog.title("Register User")
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text="Enter a name for this voice sample:",
            wraplength=380
        ).pack(pady=10)

        name_frame = ttk.Frame(dialog)
        name_frame.pack(pady=5)

        ttk.Label(name_frame, text="User Name:").pack(side=tk.LEFT, padx=5)
        name_entry = ttk.Entry(name_frame, width=30)
        name_entry.pack(side=tk.LEFT, padx=5)
        name_entry.focus()

        # Add checkbox for data augmentation
        augment_var = tk.BooleanVar(value=True)
        augment_check = ttk.Checkbutton(
            dialog,
            text="Create augmented samples (recommended)",
            variable=augment_var
        )
        augment_check.pack(pady=5)

        # Add note about multiple samples
        ttk.Label(
            dialog,
            text="Note: Recording multiple samples (5+) improves recognition accuracy",
            font=("Arial", 9),
            foreground="blue",
            wraplength=380
        ).pack(pady=5)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(
            button_frame,
            text="Register",
            command=lambda: self.controller.register_user(audio, sample_rate, name_entry.get().strip(),
                                                          augment_var.get())
            if name_entry.get().strip() else messagebox.showwarning("Warning", "Please enter a name for the user")
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.LEFT, padx=5)

    def update_user_list(self, users, sample_counts=None):
        """Update the list of registered users"""
        # Clear the list
        for item in self.users_list.get_children():
            self.users_list.delete(item)

        # Add users to the list with sample counts
        for user in users:
            count = sample_counts.get(user, 0) if sample_counts else 0
            self.users_list.insert("", tk.END, values=(user, f"{count} samples"))

    def update_listening_status(self, is_listening):
        """Update the listening status"""
        if is_listening:
            self.listening_label.config(text="Listening", foreground="green")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.progress_bar.start()

            # Add system message to chat
            self.add_system_message("Listening started. Speak now...")
        else:
            self.listening_label.config(text="Not listening", foreground="red")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress_bar.stop()

    def add_system_message(self, message):
        """Add a system message to the chat"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"System: {message}\n\n", "system")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def add_speech_message(self, speaker, text, confidence):
        """Add a speech message to the chat"""
        self.chat_text.config(state=tk.NORMAL)

        # Add speaker info
        self.chat_text.insert(tk.END, f"Speaker Identified: {speaker}\n", "header")

        # Add speech text
        self.chat_text.insert(tk.END, f"Says: ", "header")
        self.chat_text.insert(tk.END, f"{text}\n", "speech")

        # Add confidence level with appropriate color
        if confidence >= 70:
            confidence_tag = "confidence_high"
        elif confidence >= 40:
            confidence_tag = "confidence_medium"
        else:
            confidence_tag = "confidence_low"

        self.chat_text.insert(tk.END, f"Confidence: {confidence}%\n\n", confidence_tag)

        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)