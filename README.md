# Smart Mirror with Raspberry Pi
## Overview
This project implements a smart mirror powered by Raspberry Pi that uses computer vision and speech recognition to provide personalized outfit recommendations based on the user's appearance, speech input, and current weather conditions.

## Features
- **Voice Activation**: Responds to activation keywords for hands-free operation
- **Speech Recognition**: Uses offline speech-to-text processing via WhisperAI
- **User Image Analysis**: Captures and processes images with privacy-preserving face blurring
- **Weather Integration**: Provides real-time weather data for contextual recommendations
- **AI-Powered Recommendations**: Leverages Gemini AI to generate outfit suggestions
- **Text-to-Speech Output**: Delivers recommendations through voice responses

## Hardware Requirements
- Raspberry Pi (4 recommended)
- Display monitor with two-way mirror overlay
- PiCamera module or USB webcam
- Microphone (or ESP32 as audio recorder)
- Speakers for audio output
- Power supply and appropriate cables

## Software Dependencies
- Python 3.9+
- OpenCV
- TensorFlow/PyTorch
- WhisperAI for speech recognition
- Pyttsx3 for text-to-speech
- Google Gemini API for AI recommendations
- OpenWeather API for weather data

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/likhithayinala/embedded-ai-project.git
   cd embedded-ai-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Create a file `data/api_keys.txt` with your API keys:
     ```
     YOUR_OPENWEATHER_API_KEY
     YOUR_GEMINI_API_KEY
     ```

## Usage

1. Start the main application:
   ```bash
   python src/main.py
   ```

2. The system will:
   - Train a voice activation model if running for the first time
   - Ask for your location and wardrobe information
   - Listen for activation keywords
   - Capture and process your image (with privacy-preserving face blurring)
   - Provide outfit recommendations based on your appearance and the weather

## Project Structure
```
embedded-ai-project/
├── data/                    # Configuration files and trained models
│   └── api_keys.txt         # API keys for external services
├── results/                 # Generated session results and logs
├── src/                     # Source code
│   ├── main.py              # Main application entry point
│   ├── audio.py             # Audio processing for ESP32
│   └── fine_tune.py         # Model training utilities
└── README.md                # Project documentation
```

## Custom Hardware Setup
This project can use an ESP32 as an external audio recorder. Connect the ESP32 via USB and ensure it's programmed with the appropriate firmware to send audio data over serial.

## Contribution
Contributions to this project are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
[MIT License](LICENSE)

## Acknowledgements
- [WhisperAI](https://github.com/openai/whisper) for offline speech recognition
- [Google Gemini API](https://ai.google.dev/gemini-api) for AI-powered recommendations
- [OpenWeather API](https://openweathermap.org/api) for weather data
