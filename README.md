# Neuryx - Local Neural Speech Engine 🧠🎙️

Neuryx is a powerful **native desktop application** for real-time, offline speech transcription. It processes audio entirely on your local machine using customized AI models, ensuring 100% privacy and zero latency issues from network dependencies.

Built with **Python (FastAPI + Faster-Whisper)** backend and **React (Vite)** frontend, packaged as a standalone desktop app via **pywebview**.

---

## ✨ Features

- **🚀 Native Desktop App**: Runs in its own window (no browser tabs!).
- **🔒 100% Offline & Private**: Audio never leaves your computer.
- **⚡ Real-time Streaming**: Transcribes as you speak with minimal latency.
- **🌍 Multi-Language Support**:
  - **English**: High accuracy with context awareness.
  - **Urdu (اردو)**: Native script support.
  - **Roman Urdu**: "Desi style" transcription (e.g., "Kya haal hai?").
- **🏎️ Performance Optimized**:
  - Runs on modest hardware (low RAM usage with `small` model).
  - Smart throttling and memory management for endless sessions without slowdown.
- **🎨 Premium UI**: Glassmorphism dark theme, intuitive design.

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Node.js & npm

### 1. Setup Backend

```bash
# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Frontend

```bash
cd frontend
npm install
npm run build  # Compiles the React app to static files
cd ..
```

---

## 🚀 How to Run

Just double-click **`Neuryx.bat`** in the project root!

Or from terminal:

```bash
.\Neuryx.bat
```

This will:

1. Start the backend server silently.
2. Launch the native desktop window.
3. Automatically load the AI model (faster startup).

---

## 🏗️ Architecture

- **Launcher (`launcher.py`)**: Entry point. Starts Uvicorn in a daemon thread and opens the PyWebview native window.
- **Backend (`backend/`)**: FastAPI server handling audio streaming via WebSockets and running Faster-Whisper inference.
- **Frontend (`frontend/`)**: React + TypeScript UI, communicating with backend via WebSocket (`/stream`).

## 🔧 Troubleshooting

- **"Model loading..." takes time**: The first time you run it, it downloads the model (~500MB). Subsequent runs are faster.
- **Microphone issues**: Ensure your default system microphone is set correctly before launching.
