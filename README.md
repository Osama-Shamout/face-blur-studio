# 🎥 Face Blur Studio

> **Real-time, privacy-focused camera processor with zero OBS required.**

**Face Blur Studio** is a high-performance desktop application that detects faces in real-time camera feeds and applies customizable blurring, pixelation, or solid masking. It outputs directly to a virtual camera endpoint, making it instantly compatible with **Discord, Zoom, Microsoft Teams, and WebRTC web apps**.

---

## ✨ Key Features

| Category | Features |
| :--- | :--- |
| **Detection** | • Powered by **YuNet ONNX** with automatic OpenCV Haar Cascade fallback.<br>• Adjustable detection threshold and face padding. |
| **Effects** | • **Modes:** Gaussian Blur, Pixelate, and Solid Masking.<br>• **Geometries:** Soft-blended Ellipse or Sharp Rectangle. |
| **Image Control** | • Live brightness & contrast adjustments.<br>• Frame mirroring toggle. |
| **Virtual Output** | • Native integration with **Unity Video Capture** (`pyvirtualcam`).<br>• Exposes 640x480 @ 30 FPS stream optimized for WebRTC app compatibility. |
| **Performance** | • Smart frame-skipping algorithm for detection.<br>• Off-thread PPM rendering engine for zero-lag UI previewing. |
| **UI / UX** | • Modern Tkinter interface with smooth color transitions.<br>• Toggleable **Dark / Light Theme**. |

---

## 🛠️ System Requirements

* **OS:** Windows 10 / 11 (x64)
* **Python:** Version 3.10 or higher
* **Camera:** Integrated Webcam, USB Camera, or Virtual Input (e.g., Iriun Webcam)

---

## 📦 Installation & Setup

1. **Clone or Download** the repository to your local folder.
2. **Connect** your physical or virtual camera device.
3. **Install Dependencies** by opening a terminal in the project directory and running:

```cmd
pip install -r requirements.txt