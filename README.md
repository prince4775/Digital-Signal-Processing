# 🎧 Digital Signal Processing Toolkit

*A Streamlit-based interactive environment for visualizing, processing, and analyzing digital signals.*

## 📌 Overview

This repository provides an **interactive DSP application** built using **Python** and **Streamlit**, allowing students, researchers, and engineers to experiment with real-time audio processing.
The tool is designed for **Digital Signal Processing laboratory work**, enabling users to upload audio, visualize signals, add noise, apply filters, and analyze spectrums effortlessly.

---

## 🚀 Key Features

### 🎙 Audio Input

* Upload audio files (WAV, MP3)
* Record voice directly from the browser
* Display audio properties (sampling rate, duration, channels)

### 📊 Signal Visualization

* Time-domain waveform
* Frequency-domain magnitude spectrum using FFT
* High-frequency vs low-frequency comparisons
* Interactive and real-time updates

### 🛠 Noise Processing

* Add configurable **random noise** to clean audio
* Noise removal using:

  * **Wiener Filter** (statistical noise reduction)
  * (Optional) Low-pass and High-pass filters
* Play clean, noisy, and filtered audio

### 🎛 DSP Operations

* Filtering with customizable cutoff frequency
* SNR (Signal-to-Noise Ratio) calculation
* Spectrogram generation
* Real-time analysis through Streamlit UI

---

## 🧠 How It Works

The DSP app uses:

* **NumPy** for signal manipulation
* **SciPy** for filtering operations
* **Streamlit** for front-end interface
* **Matplotlib / Plotly** for visualizations
* **Librosa / Soundfile** for audio handling

Workflow:

1. User uploads or records an audio signal
2. The system extracts sampling rate and samples
3. The waveform and spectrum are visualized
4. User can add random noise
5. Noise is removed through the **Wiener filter**
6. App displays and plays the processed audio


## 📘 Technologies Used

* **Python**
* **Streamlit**
* **NumPy**
* **SciPy**
* **Librosa**
* **Matplotlib / Plotly**

---

## 📚 DSP Concepts Used

* Fourier Transform
* Frequency Analysis
* Noise Generation
* Wiener Filtering
* Digital Filtering (LPF, HPF, BPF)
* Spectrograms


## 🤝 Contributing

Contributions are welcome!
Feel free to submit issues, feature requests, or pull requests to enhance the project.

---

## 🧑‍💻 Author

**Muhammad Ayub**
Electrical Engineering – DSP & Machine Learning
