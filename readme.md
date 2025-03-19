# MultSpeak Real-time Speaker Recognition System

MultSpeak is an advanced real-time speaker recognition and transcription system that can identify who is speaking and
what they are saying in real-time.

## Features

- **User Registration**: Add multiple voice samples for different users
- **Real-time Recognition**: Continuously listen and identify speakers
- **Speech Transcription**: Convert speech to text in real-time
- **Speaker Identification**: Match voice patterns to known users
- **Confidence Scoring**: Display confidence levels for speaker identification

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```
zz
## How It Works

1. **Setup Phase**: Add voice samples for each user that will be part of the conversation
2. **Real-time Recognition**: The system continuously listens to the microphone input
3. **Speech Analysis**: When someone speaks, the system:
    - Transcribes the speech to text
    - Extracts voice features
    - Identifies the speaker
    - Displays results with confidence levels

## Usage Workflow

1. Start the application
2. Add voice samples for all users by clicking "Record Voice Sample" or "Upload Voice Sample"
3. Give each sample a name when prompted
4. Click "Start Listening" to begin real-time recognition
5. Speak normally - the system will identify speakers and transcribe speech automatically
6. Click "Stop Listening" when done

## Requirements

- Python 3.7+
- Microphone
- Internet connection (for Google Speech Recognition API)

## Technical Details

The system uses:

- **MFCCs (Mel-frequency cepstral coefficients)** for voice feature extraction
- **Google Speech Recognition API** for speech-to-text conversion
- **Cosine similarity** for speaker identification
- **Multi-threading** for real-time processing