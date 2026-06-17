# 🎥 Video Transcription App

A powerful Streamlit application that converts videos to text with automatic speech-to-text transcription using OpenAI Whisper.

## ✨ Features

- **Video Upload**: Support for MP4, MOV, AVI, MKV, WMV formats
- **Audio Extraction**: Automatic audio extraction from videos
- **Smart Chunking**: Splits audio into manageable 30-second chunks
- **Automatic Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **Multi-language Support**: Automatically detects and transcribes Arabic, English, and 50+ other languages
- **Time-based File Naming**: Creates transcription files named by time (0.txt, 30.txt, 60.txt, etc.)
- **Complete Download Kit**: Includes audio chunks, transcriptions, and templates
- **Progress Tracking**: Real-time progress updates during processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

### Installation

1. **Clone or copy the project folder** to your computer

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv transcription_env
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     transcription_env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source transcription_env/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install FFmpeg** (required for video processing):
   - **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload a video file** using the file uploader
2. **Adjust chunk duration** if needed (default: 30 seconds)
3. **Click "🚀 Process Video"** to start processing
4. **Wait for transcription** to complete (may take a few minutes)
5. **Download the complete kit** containing:
   - Audio chunks (30-second segments)
   - Transcription files (0.txt, 30.txt, 60.txt, etc.)
   - Full transcription (full_transcription.txt)
   - Templates for manual transcription

## 📁 Project Structure

```
video-transcription-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── audio_chunks/             # Generated audio chunks (created during processing)
├── templates/                # Transcription templates (created during processing)
└── transcriptions/           # Transcription files (created during processing)
```

## 🔧 Dependencies

### Core Dependencies
- **streamlit**: Web application framework
- **openai-whisper**: Speech-to-text transcription
- **torch/torchvision/torchaudio**: Machine learning framework (PyTorch)

### Optional Fallback Dependencies
- **moviepy**: Alternative video processing
- **pydub**: Alternative audio processing
- **ffmpeg-python**: FFmpeg Python bindings

### System Requirements
- **FFmpeg**: Required for video/audio processing
- **Python 3.8+**: Minimum Python version

## 🌍 Language Support

The app automatically detects and transcribes in multiple languages:
- Arabic (العربية)
- English
- French (Français)
- German (Deutsch)
- Spanish (Español)
- And 50+ other languages supported by Whisper

## 🐛 Troubleshooting

### Common Issues

1. **"FFmpeg not found" error**:
   - Install FFmpeg and add to system PATH
   - Restart the application

2. **PyTorch installation issues**:
   - The app uses CPU-only PyTorch for compatibility
   - If issues persist, try: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

3. **Whisper model download**:
   - First run may take longer as it downloads the model
   - Ensure stable internet connection

4. **Memory issues**:
   - For very long videos, consider shorter chunk durations
   - Close other applications to free up RAM

### Performance Tips

- **Use shorter chunks** (15-30 seconds) for faster processing
- **Close unnecessary applications** to free up system resources
- **Use good quality audio** for better transcription accuracy
- **Process videos sequentially** rather than in parallel

## 📝 File Naming Convention

Transcription files are named by their starting time:
- `0.txt` - First 30 seconds (0:00 - 0:30)
- `30.txt` - Second 30 seconds (0:30 - 1:00)
- `60.txt` - Third 30 seconds (1:00 - 1:30)
- And so on...

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting features
- Submitting pull requests

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify FFmpeg is installed and accessible
4. Try restarting the application

---

**Happy transcribing! 🎉**
