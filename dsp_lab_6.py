# dsp_lab_streamlit_freq_z.py

import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from io import BytesIO
from scipy.signal import wiener

# ----------------------
# Streamlit Page Setup
# ----------------------
st.set_page_config(page_title="DSP Audio Freq & Z Domain App", layout="wide")

# ---- Header ----
st.markdown("""
### **Muhammad Ayub — Reg No: 22jzele0470**
## DSP Audio Frequency & Z-Domain App
""")

# ----------------------
# Helper Functions
# ----------------------
def ensure_folder(path="recordings"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_wav(filename, audio, fs):
    sf.write(filename, audio, fs)

def audio_bytes_from_array(arr, fs):
    buf = BytesIO()
    sf.write(buf, arr, fs, format="WAV")
    buf.seek(0)
    return buf.read()

def plot_spectrum(signal, fs, title="Spectrum"):
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs[:len(freqs)//2], fft_vals[:len(freqs)//2])
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    plt.tight_layout()
    return fig

def plot_z_domain(signal, title="Z-Domain Plot"):
    # Z-domain using unit circle representation
    z = np.fft.fft(signal)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(np.real(z), np.imag(z), 'o')
    ax.set_title(title)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    plt.tight_layout()
    return fig

def play_audio(audio, fs):
    sd.play(audio, fs)

def stop_audio():
    sd.stop()

# ----------------------
# UI Controls
# ----------------------
col1, col2 = st.columns([2, 1])
with col1:
    duration = st.slider("Recording duration (seconds)", 1, 10, 3)
    fs = st.selectbox("Sampling rate (Hz)", [16000, 22050, 44100], index=2)
    st.write("Ensure your microphone is enabled.")

with col2:
    save_dir = ensure_folder("recordings")
    st.write("Saved recordings folder:")
    st.write(save_dir)

# ----------------------
# Record Audio
# ----------------------
if st.button("🎤 Record Audio"):
    st.info("Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    ts = int(time.time())
    clean_path = os.path.join(save_dir, f"clean_{ts}.wav")
    save_wav(clean_path, audio, fs)

    st.session_state["audio_clean"] = audio
    st.session_state["audio_clean_path"] = clean_path

    st.success(f"Recording complete — saved to {clean_path}")
    st.audio(audio_bytes_from_array(audio, fs), format="audio/wav")

    # Plots after recording
    st.markdown("### Frequency Domain")
    st.pyplot(plot_spectrum(audio, fs, "Clean Audio Spectrum"))

    st.markdown("### Z-Domain")
    st.pyplot(plot_z_domain(audio, "Clean Audio Z-Domain"))

# ----------------------
# Noise Section
# ----------------------
if "audio_clean" in st.session_state:
    st.subheader("Add Noise Section")
    colA, colB = st.columns(2)
    with colA:
        noise_mode = st.selectbox("Noise Type", ["White Gaussian", "Uniform"], index=0)
        noise_level = st.slider("Noise strength (std. dev.)", 0.0, 0.2, 0.02, step=0.001)

    with colB:
        snr_db = st.slider("Target SNR (dB) — 0 = ignore", -10, 40, 0)
        use_target_snr = snr_db != 0

    if st.button("➕ Add Noise"):
        clean = st.session_state["audio_clean"]

        if use_target_snr:
            sig_power = np.mean(clean**2)
            target_power = sig_power / (10**(snr_db/10))
            noise = np.random.randn(len(clean))
            noise = noise / np.std(noise) * np.sqrt(target_power)
        else:
            if noise_mode == "White Gaussian":
                noise = np.random.normal(0.0, noise_level, len(clean))
            else:
                noise = np.random.uniform(-noise_level, noise_level, len(clean))

        noisy = clean + noise
        ts = int(time.time())
        noisy_path = os.path.join(save_dir, f"noisy_{ts}.wav")
        save_wav(noisy_path, noisy, fs)

        st.session_state["audio_noisy"] = noisy
        st.session_state["audio_noisy_path"] = noisy_path
        st.session_state["added_noise"] = noise

        st.success(f"Noisy audio saved to {noisy_path}")
        st.audio(audio_bytes_from_array(noisy, fs), format="audio/wav")

        # Plots
        st.markdown("### Frequency Domain — Noisy Audio")
        st.pyplot(plot_spectrum(noisy, fs, "Noisy Audio Spectrum"))

        st.markdown("### Z-Domain — Noisy Audio")
        st.pyplot(plot_z_domain(noisy, "Noisy Audio Z-Domain"))

# ----------------------
# Noise Removal (Wiener Filter)
# ----------------------
if "audio_noisy" in st.session_state:
    st.subheader("Noise Removal Section")
    if st.button("🧹 Remove Noise (Wiener Filter)"):
        noisy = st.session_state["audio_noisy"]
        denoised = wiener(noisy, mysize=29)
        st.session_state["audio_denoised"] = denoised

        ts = int(time.time())
        denoised_path = os.path.join(save_dir, f"denoised_{ts}.wav")
        save_wav(denoised_path, denoised, fs)
        st.session_state["audio_denoised_path"] = denoised_path

        st.success(f"Noise removed — saved to {denoised_path}")
        st.audio(audio_bytes_from_array(denoised, fs), format="audio/wav")

        st.markdown("### Frequency Domain — Denoised Audio")
        st.pyplot(plot_spectrum(denoised, fs, "Denoised Audio Spectrum"))

        st.markdown("### Z-Domain — Denoised Audio")
        st.pyplot(plot_z_domain(denoised, "Denoised Audio Z-Domain"))

# ----------------------
# Download Buttons
# ----------------------
st.markdown("### Download Audio Files")

c1, c2, c3 = st.columns(3)
clean_path = st.session_state.get("audio_clean_path")
noisy_path = st.session_state.get("audio_noisy_path")
denoised_path = st.session_state.get("audio_denoised_path")

with c1:
    if clean_path:
        with open(clean_path, "rb") as f:
            st.download_button("Download Clean Audio", f, file_name=os.path.basename(clean_path))

with c2:
    if noisy_path:
        with open(noisy_path, "rb") as f:
            st.download_button("Download Noisy Audio", f, file_name=os.path.basename(noisy_path))

with c3:
    if denoised_path:
        with open(denoised_path, "rb") as f:
            st.download_button("Download Denoised Audio", f, file_name=os.path.basename(denoised_path))
