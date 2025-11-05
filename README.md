# Clash Royale Pocket Companion - Community Edition

> **Work in Progress**: This project is currently unfinished. Card detections and elixir calculations may be inaccurate. Expect bugs and ongoing improvements.

**Open-source desktop assistant for tracking Clash Royale matches in real time. This application best works for players in Ranked mode where spectators cannot view the elixir count.**

## Demo Sneak Peek

https://github.com/user-attachments/assets/c873d76a-9abd-42ce-88b9-3ff93cfa556b

## Features

**Real-Time Match Tracking**
- Track elixir counts for your opponent
- Monitor cards played during matches
- Match timer with OCR detection

**Computer Vision Powered**
- Automatic card detection using Resnet
- PaddleOCR for timer and slot detection
- Template matching for VS badge detection
- Real-time frame capture from Clash Royale window

**Desktop Application**
- Modern Electron + React frontend
- PyQt6 backend for computer vision

## Project Structure

- `frontend/` – Electron + React desktop app
- `backend/` – PyQt6 application with CV pipelines
- `assets/` – Card imagery, templates, weights, and embeddings

## Quick Start

### Prerequisites

- **Python 3.10+** (Windows recommended for capture APIs)
- **Node.js 18+** (for Electron tooling)
- **Git** (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chanejianrungsang/royalecompanion-community.git
   cd royalecompanion-community
   ```

2. **Set up the backend**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows PowerShell
   # On macOS/Linux: source .venv/bin/activate
   
   pip install -r backend/requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

**Option 1: Run the full desktop app**
```bash
cd frontend
npm run dev
```
This starts both the Vite dev server and Electron with hot reload.

**Option 2: Run backend GUI separately** (for testing CV)
```bash
.venv\Scripts\activate
python backend/main.py
```

**Option 3: Run headless backend** (JSON bridge for Electron)
```bash
.venv\Scripts\activate
python backend/main_headless.py
```

## Tech Stack

**Frontend:**
- Electron - Desktop app framework
- React 19 - UI library
- Vite - Build tool
- TailwindCSS - Styling
- Zustand - State management

**Backend:**
- PyQt6 - GUI framework
- OpenCV - Computer vision
- PaddleOCR - Text recognition
- Resnet - Card Detection
- NumPy - Numerical computing

## Known Issues & Limitations

- Windows is recommended due to screen capture APIs
- Requires a different Clash Royale instance 
- Initial calibration required for accurate card detection
- Timer OCR may require adjustment based on screen resolution

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Clash Royale is a trademark of Supercell
- This is an unofficial project, not affiliated with Supercell

