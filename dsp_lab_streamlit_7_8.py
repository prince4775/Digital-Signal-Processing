# dsp_lab_streamlit.py

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
st.set_page_config(page_title="DSP Audio Recording & Noise App", layout="wide")

# ---- Top Header with Picture + Name ----
col_logo, col_text = st.columns([1, 3])

with col_logo:
    st.image("ayub.png", width=230)

with col_text:
    st.markdown("""
    ### **Muhammad Ayub — Reg No: 22jzele0470**
    ## DSP Audio Recording and Noise Addition App
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

def plot_waveform(signal, title="Waveform"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrum(signal, fs, title="Spectrum"):
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/fs)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, fft_vals)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(signal, fs, title="Spectrogram (Heat Map)"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.specgram(signal, Fs=fs, cmap="inferno")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    return fig

# Play / pause helpers
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
    if "recordings" not in st.session_state:
        st.session_state["recordings"] = {}
    save_dir = ensure_folder("recordings")
    st.write("Saved recordings folder:")
    st.write(save_dir)

# ----------------------
# Record Audio Button
# ----------------------
if st.button("Record Speech"):
    try:
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

        # Playback buttons
        play_col, pause_col = st.columns(2)
        if play_col.button("▶ Play Clean Audio"):
            play_audio(audio, fs)
        if pause_col.button("⏸ Pause Audio"):
            stop_audio()

        # Time-Domain
        st.markdown("### Clean Signal — Time Domain")
        st.pyplot(plot_waveform(audio, "Clean Speech Signal"))

        # Frequency-Domain
        st.markdown("### Clean Signal — Frequency Domain")
        st.pyplot(plot_spectrum(audio, fs, "Clean Speech Spectrum"))

        # Spectrogram
        st.markdown("### Clean Signal — Spectrogram (Heat Map)")
        st.pyplot(plot_spectrogram(audio, fs, "Clean Signal Spectrogram"))

    except Exception as e:
        st.error(f"Recording failed: {e}")

st.markdown("---")

# -----------------------------------------------------------------------
# Noise Section
# -----------------------------------------------------------------------
if "audio_clean" in st.session_state:
    st.subheader("Add Noise / Playback / Analysis Window")

    colA, colB, colC = st.columns(3)

    with colA:
        noise_mode = st.selectbox("Noise Type", ["White Gaussian", "Uniform"], index=0)
        noise_level = st.slider("Noise strength (std. dev.)", 0.0, 0.2, 0.02, step=0.001)

    with colB:
        snr_db = st.slider("Target SNR (dB) — (0 = ignore)", -10, 40, 0)
        use_target_snr = snr_db != 0

    with colC:
        if st.button("Add Noise"):
            clean = st.session_state["audio_clean"]

            # Generate noise
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
            st.session_state["added_noise"] = noise  # store the added noise

            st.success(f"Noisy audio saved to {noisy_path}")
            st.audio(audio_bytes_from_array(noisy, fs), format="audio/wav")

            # Playback buttons
            playN, pauseN = st.columns(2)
            if playN.button("▶ Play Noisy Audio"):
                play_audio(noisy, fs)
            if pauseN.button("⏸ Pause Noisy Audio"):
                stop_audio()

            st.markdown("## Clean vs Noisy Signal — Full Analysis Window")

            # ---------------------------
            # TIME DOMAIN
            # ---------------------------
            t1, t2 = st.columns(2)
            with t1:
                st.pyplot(plot_waveform(clean, "Clean Signal — Time Domain"))
            with t2:
                st.pyplot(plot_waveform(noisy, "Noisy Signal — Time Domain"))

            # ---------------------------
            # FREQUENCY DOMAIN
            # ---------------------------
            f1, f2 = st.columns(2)
            with f1:
                st.pyplot(plot_spectrum(clean, fs, "Clean Signal — Spectrum"))
            with f2:
                st.pyplot(plot_spectrum(noisy, fs, "Noisy Signal — Spectrum"))

            # ---------------------------
            # SPECTROGRAMS (HEAT MAP)
            # ---------------------------
            s1, s2 = st.columns(2)
            with s1:
                st.pyplot(plot_spectrogram(clean, fs, "Clean Signal — Spectrogram"))
            with s2:
                st.pyplot(plot_spectrogram(noisy, fs, "Noisy Signal — Spectrogram"))

# -----------------------------------------------------------------------
# Noise Removal Section (Wiener Filter)
# -----------------------------------------------------------------------
if "audio_noisy" in st.session_state:
    st.subheader("Noise Removal / Wiener Filter Section")

    if st.button("Remove Noise (Wiener Filter)"):
        noisy = st.session_state["audio_noisy"]
        # Apply Wiener filter
        denoised = wiener(noisy, mysize=29)
        st.session_state["audio_denoised"] = denoised

        # Save denoised audio
        ts = int(time.time())
        denoised_path = os.path.join(save_dir, f"denoised_{ts}.wav")
        save_wav(denoised_path, denoised, fs)
        st.session_state["audio_denoised_path"] = denoised_path

        st.success(f"Noise removed — saved to {denoised_path}")
        st.audio(audio_bytes_from_array(denoised, fs), format="audio/wav")

        # ---------------------------
        # Plots for Denoised Signal
        # ---------------------------
        st.markdown("### Denoised Signal — Time Domain")
        st.pyplot(plot_waveform(denoised, "Denoised Signal — Time Domain"))

        st.markdown("### Denoised Signal — Frequency Domain")
        st.pyplot(plot_spectrum(denoised, fs, "Denoised Signal — Spectrum"))

        st.markdown("### Denoised Signal — Spectrogram (Heat Map)")
        st.pyplot(plot_spectrogram(denoised, fs, "Denoised Signal — Spectrogram"))

# ----------------------
# Download Buttons
# ----------------------
st.markdown("### Saved Files (Evidence)")

c1, c2, c3 = st.columns(3)
clean_path = st.session_state.get("audio_clean_path")
noisy_path = st.session_state.get("audio_noisy_path")
denoised_path = st.session_state.get("audio_denoised_path")

with c1:
    if clean_path:
        with open(clean_path, "rb") as f:
            st.download_button("Download Clean WAV", f, file_name=os.path.basename(clean_path))

with c2:
    if noisy_path:
        with open(noisy_path, "rb") as f:
            st.download_button("Download Noisy WAV", f, file_name=os.path.basename(noisy_path))

with c3:
    if denoised_path:
        with open(denoised_path, "rb") as f:
            st.download_button("Download Denoised WAV", f, file_name=os.path.basename(denoised_path))
