# dsp_lab_streamlit.py
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from io import BytesIO

st.set_page_config(page_title="DSP Audio Recording & Noise App", layout="wide")

col1, col2 = st.columns([1, 3])  # 1:3 ratio for pic vs text

with col1:
    st.image("ayub.png", width=220)  # replace "my_pic.jpg" with your filename

with col2:
    st.markdown("""
    # **Muhammad Ayub**
    # **Reg No: 22jzele0470**
    """) 

st.title("DSP Audio Recording and Noise Addition App")

# --- Helpers ---
def ensure_folder(path="recordings"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_wav(filename, audio, fs):
    # audio expected as 1-D float32/float64 numpy array in range [-1,1] or similar
    sf.write(filename, audio, fs)  # soundfile handles floats well

def plot_waveform(signal, title="Waveform"):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(signal)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def audio_bytes_from_array(arr, fs):
    # Convert numpy float array to WAV bytes (in-memory) for st.audio
    buf = BytesIO()
    sf.write(buf, arr, fs, format="WAV")
    buf.seek(0)
    return buf.read()

# --- UI controls ---
col1, col2 = st.columns([2,1])

with col1:
    duration = st.slider("Recording duration (seconds)", 1, 10, 3)
    fs = st.selectbox("Sampling rate (Hz)", [16000, 22050, 44100], index=2)
    st.write("Microphone test: make sure your mic is allowed in Windows settings or browser if necessary.")

with col2:
    if "recordings" not in st.session_state:
        st.session_state["recordings"] = {}
    save_dir = ensure_folder("recordings")
    st.write("Saved recordings folder:")
    st.write(save_dir)

# --- Record button & logic ---
if st.button("Record Speech"):
    try:
        st.info("Recording... Speak now.")
        # Record (blocking)
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        ts = int(time.time())
        clean_path = os.path.join(save_dir, f"clean_{ts}.wav")
        save_wav(clean_path, audio, fs)

        # Save to session
        st.session_state["audio_clean"] = audio
        st.session_state["audio_clean_path"] = clean_path

        st.success(f"Recording complete — saved to {clean_path}")
        st.audio(audio_bytes_from_array(audio, fs), format="audio/wav")  # in-app player
        st.pyplot(plot_waveform(audio, title="Clean Speech Signal"))
    except Exception as e:
        st.error(f"Recording failed: {e}")

st.markdown("---")

# --- Noise controls (only shown if we have audio) ---
if "audio_clean" in st.session_state:
    st.subheader("Add Noise / Playback / Save evidence")
    colA, colB, colC = st.columns(3)

    with colA:
        noise_mode = st.selectbox("Noise Type", ["White Gaussian", "Uniform"], index=0)
        noise_level = st.slider("Noise strength (std. dev.)", 0.0, 0.2, 0.02, step=0.001)

    with colB:
        snr_db = st.slider("Target SNR (dB) — optional (0 = off)", -10, 40, 0)
        use_target_snr = snr_db != 0

    with colC:
        if st.button("Add Noise"):
            clean = st.session_state["audio_clean"]
            if use_target_snr:
                # compute noise for given SNR
                sig_power = np.mean(clean**2)
                target_power = sig_power / (10**(snr_db/10))
                noise = np.random.randn(len(clean)).astype('float32')  # unit variance
                noise = noise / np.std(noise) * np.sqrt(target_power)
            else:
                if noise_mode == "White Gaussian":
                    noise = np.random.normal(0.0, noise_level, size=len(clean)).astype('float32')
                else:
                    noise = np.random.uniform(-noise_level, noise_level, size=len(clean)).astype('float32')

            noisy = clean + noise
            ts = int(time.time())
            noisy_path = os.path.join(save_dir, f"noisy_{ts}.wav")
            save_wav(noisy_path, noisy, fs)

            st.session_state["audio_noisy"] = noisy
            st.session_state["audio_noisy_path"] = noisy_path

            st.success(f"Noisy audio saved to {noisy_path}")
            st.audio(audio_bytes_from_array(noisy, fs), format="audio/wav")
            st.pyplot(plot_waveform(noisy, title="Noisy Speech Signal"))

    # Download buttons & file paths (evidence)
    st.markdown("**Saved files (evidence)**")
    clean_path = st.session_state.get("audio_clean_path", None)
    noisy_path = st.session_state.get("audio_noisy_path", None)

    c1, c2 = st.columns(2)
    with c1:
        if clean_path:
            st.write("Clean file:")
            st.write(clean_path)
            with open(clean_path, "rb") as f:
                st.download_button("Download Clean WAV", data=f, file_name=os.path.basename(clean_path), mime="audio/wav")
    with c2:
        if noisy_path:
            st.write("Noisy file:")
            st.write(noisy_path)
            with open(noisy_path, "rb") as f2:
                st.download_button("Download Noisy WAV", data=f2, file_name=os.path.basename(noisy_path), mime="audio/wav")

    st.markdown("---")
    st.write("To provide evidence: take screenshots of the Streamlit page showing (1) the audio players and (2) the waveform plots. Also attach the saved WAV files from the `recordings/` folder when you submit.")
else:
    st.info("Record audio first to enable noise addition and evidence saving.")