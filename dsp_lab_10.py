# dsp_lab_streamlit.py

import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from io import BytesIO
from scipy.ndimage import gaussian_filter1d
import tempfile

# ----------------------
# Streamlit Page Setup
# ----------------------
st.set_page_config(page_title="DSP Audio Lab: Record/Upload + Gaussian Denoising", layout="wide")

# ---- Header ----
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("ayub.png", width=200)
with col_text:
    st.markdown("""
    ### **Muhammad Ayub — Reg No: 22jzele0470**
    # DSP Audio Processing Lab
    ### Record or Upload Speech → Add Noise → **Gaussian Filter Denoising**
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
    return buf

def plot_waveform(signal, title="Waveform", color='blue'):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal, color=color)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
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
    ax.set_xlim(0, fs//2)
    plt.tight_layout()
    return fig

def plot_spectrogram(signal, fs, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 3))
    Pxx, freqs, bins, im = ax.specgram(signal, Fs=fs, cmap="magma", NFFT=512)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax, label="Intensity (dB)")
    plt.tight_layout()
    return fig

def play_audio(audio, fs):
    sd.play(audio, fs)
    sd.wait()

def stop_audio():
    sd.stop()

# Ensure recordings folder
save_dir = ensure_folder("recordings")

# ----------------------
# Input Method Selection
# ----------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input:", ["Record Live", "Upload Audio File"])

fs = 44100  # Default sampling rate
audio_clean = None
audio_clean_path = None

# ======================
# 1. RECORD LIVE AUDIO
# ======================
if input_method == "Record Live":
    st.subheader("Live Microphone Recording")
    col1, col2 = st.columns([2, 1])
    with col1:
        duration = st.slider("Recording Duration (seconds)", 1, 15, 5)
        fs = st.selectbox("Sampling Rate (Hz)", [16000, 22050, 44100, 48000], index=2)
    
    if st.button("Start Recording", type="primary"):
        with st.spinner("Recording... Speak now!"):
            try:
                audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()
                audio_clean = audio.flatten()

                ts = int(time.time())
                audio_clean_path = os.path.join(save_dir, f"clean_recorded_{ts}.wav")
                save_wav(audio_clean_path, audio_clean, fs)

                st.session_state.audio_clean = audio_clean
                st.session_state.audio_clean_path = audio_clean_path
                st.session_state.fs = fs

                st.success(f"Recorded successfully! Saved: {os.path.basename(audio_clean_path)}")
                st.audio(audio_bytes_from_array(audio_clean, fs).read(), format="audio/wav")

            except Exception as e:
                st.error(f"Recording failed: {e}")

# ======================
# 2. UPLOAD AUDIO FILE
# ======================
else:
    st.subheader("Upload Audio File (.wav or .mp3)")
    uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with st.spinner("Loading audio file..."):
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name
            tfile.close()

            try:
                audio_clean, fs = sf.read(tfile_path)
                if audio_clean.ndim > 1:
                    audio_clean = np.mean(audio_clean, axis=1)  # Convert stereo to mono

                # Normalize to float32
                if audio_clean.dtype != np.float32:
                    audio_clean = audio_clean.astype(np.float32)
                    if np.max(np.abs(audio_clean)) > 0:
                        audio_clean /= np.max(np.abs(audio_clean))

                ts = int(time.time())
                ext = ".wav" if uploaded_file.name.endswith(".wav") else ".wav"
                audio_clean_path = os.path.join(save_dir, f"clean_uploaded_{ts}.wav")
                save_wav(audio_clean_path, audio_clean, fs)

                st.session_state.audio_clean = audio_clean
                st.session_state.audio_clean_path = audio_clean_path
                st.session_state.fs = fs

                st.success(f"Uploaded & loaded: {uploaded_file.name}")
                st.audio(audio_bytes_from_array(audio_clean, fs).read(), format="audio/wav")

            except Exception as e:
                st.error(f"Error loading audio: {e}")
            finally:
                os.unlink(tfile_path)  # Clean up temp file

# ======================
# Show Plots for Clean Audio if Available
# ======================
if "audio_clean" in st.session_state:
    clean = st.session_state.audio_clean
    fs = st.session_state.get("fs", 44100)

    st.markdown("---")
    st.subheader("Step 1: Clean Audio Analysis")

    tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])

    with tab1:
        st.pyplot(plot_waveform(clean, "Original Clean Signal", 'green'))

    with tab2:
        st.pyplot(plot_spectrum(clean, fs, "Clean Spectrum"))

    with tab3:
        st.pyplot(plot_spectrogram(clean, fs, "Clean Spectrogram"))

    # ======================
    # Add Noise Section
    # ======================
    st.subheader("Step 2: Add Noise to Clean Audio")

    colA, colB = st.columns(2)
    with colA:
        noise_type = st.selectbox("Noise Type", ["White Gaussian", "Pink Noise", "Brown Noise", "Uniform"])
        noise_level = st.slider("Manual Noise Level (σ)", 0.001, 0.3, 0.05, step=0.001)
    
    with colB:
        target_snr = st.slider("Or Set Target SNR (dB)", -15, 40, 10)
        use_snr = st.checkbox("Use Target SNR (overrides manual level)", value=True)

    if st.button("Generate Noisy Version", type="secondary"):
        with st.spinner("Adding noise..."):
            if use_snr and target_snr != 0:
                signal_power = np.mean(clean ** 2)
                noise_power = signal_power / (10 ** (target_snr / 10))
                noise_std = np.sqrt(noise_power)
            else:
                noise_std = noise_level

            if noise_type == "White Gaussian":
                noise = np.random.normal(0, noise_std, len(clean))
            elif noise_type == "Uniform":
                noise = np.random.uniform(-noise_std * 3, noise_std * 3, len(clean))
            elif noise_type == "Pink Noise":
                # Simple pink noise approximation
                white = np.random.randn(len(clean))
                fft = np.fft.rfft(white)
                freqs = np.fft.rfftfreq(len(white))
                fft /= np.sqrt(freqs + 1e-6)  # 1/f
                noise = np.fft.irfft(fft)
                noise = noise / np.std(noise) * noise_std
            elif noise_type == "Brown Noise":
                noise = np.cumsum(np.random.randn(len(clean))) * noise_std * 0.1
                noise = noise - np.mean(noise)

            noisy = clean + noise

            ts = int(time.time())
            noisy_path = os.path.join(save_dir, f"noisy_{ts}.wav")
            save_wav(noisy_path, noisy, fs)

            st.session_state.audio_noisy = noisy
            st.session_state.audio_noisy_path = noisy_path
            st.session_state.added_noise = noise

            st.success("Noisy audio generated!")
            st.audio(audio_bytes_from_array(noisy, fs).read(), format="audio/wav")

    # ======================
    # Show Plots for Noisy Audio if Available
    # ======================
    if "audio_noisy" in st.session_state:
        noisy = st.session_state.audio_noisy

        st.subheader("Step 2: Noisy Audio Analysis")

        tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])

        with tab1:
            st.pyplot(plot_waveform(noisy, "Noisy Signal", 'red'))

        with tab2:
            st.pyplot(plot_spectrum(noisy, fs, "Noisy Spectrum"))

        with tab3:
            st.pyplot(plot_spectrogram(noisy, fs, "Noisy Spectrogram"))

        # ======================
        # Gaussian Denoising Section
        # ======================
        st.markdown("---")
        st.subheader("Step 3: Gaussian Filter Denoising")

        sigma = st.slider("Gaussian Filter Sigma (smoothing strength)", 1, 100, 25)

        if st.button("Apply Gaussian Denoising Filter", type="primary"):
            with st.spinner("Applying Gaussian filter..."):
                denoised = gaussian_filter1d(noisy, sigma=sigma)

                # Optional: normalize
                if np.max(np.abs(denoised)) > 0:
                    denoised = denoised / np.max(np.abs(denoised)) * 0.95

                ts = int(time.time())
                denoised_path = os.path.join(save_dir, f"denoised_gaussian_{ts}.wav")
                save_wav(denoised_path, denoised, fs)

                st.session_state.audio_denoised = denoised
                st.session_state.audio_denoised_path = denoised_path

                st.success("Gaussian denoising applied successfully!")
                st.audio(audio_bytes_from_array(denoised, fs).read(), format="audio/wav")

        # ======================
        # Show Plots for Denoised Audio if Available
        # ======================
        if "audio_denoised" in st.session_state:
            denoised = st.session_state.audio_denoised

            st.subheader("Step 3: Denoised Audio Analysis")

            tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])

            with tab1:
                st.pyplot(plot_waveform(denoised, "Gaussian Denoised Signal", 'blue'))

            with tab2:
                st.pyplot(plot_spectrum(denoised, fs, "Denoised Spectrum"))

            with tab3:
                st.pyplot(plot_spectrogram(denoised, fs, "Denoised Spectrogram"))

            # ======================
            # Comparison Section
            # ======================
            st.markdown("---")
            st.subheader("Full Comparison: Clean | Noisy | Denoised")

            comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])

            with comp_tab1:
                c1, c2, c3 = st.columns(3)
                with c1: st.pyplot(plot_waveform(clean, "Original Clean", 'green'))
                with c2: st.pyplot(plot_waveform(noisy, "Noisy Signal", 'red'))
                with c3: st.pyplot(plot_waveform(denoised, "Gaussian Denoised", 'blue'))

            with comp_tab2:
                c1, c2, c3 = st.columns(3)
                with c1: st.pyplot(plot_spectrum(clean, fs, "Clean Spectrum"))
                with c2: st.pyplot(plot_spectrum(noisy, fs, "Noisy Spectrum"))
                with c3: st.pyplot(plot_spectrum(denoised, fs, "Denoised Spectrum"))

            with comp_tab3:
                c1, c2, c3 = st.columns(3)
                with c1: st.pyplot(plot_spectrogram(clean, fs, "Clean"))
                with c2: st.pyplot(plot_spectrogram(noisy, fs, "Noisy"))
                with c3: st.pyplot(plot_spectrogram(denoised, fs, "Denoised (Gaussian)"))

# ======================
# Download Section
# ======================
st.markdown("---")
st.subheader("Download Your Files")

cols = st.columns(4)
paths = [
    st.session_state.get("audio_clean_path"),
    st.session_state.get("audio_noisy_path"),
    st.session_state.get("audio_denoised_path")
]
labels = ["Clean Audio", "Noisy Audio", "Denoised (Gaussian)"]
icons = ["🟢", "🔴", "🔵"]

for col, path, label, icon in zip(cols, paths + [None], labels + ["Help"], icons + ["❓"]):
    with col:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    label=f"{icon} {label}",
                    data=f,
                    file_name=os.path.basename(path),
                    mime="audio/wav"
                )
        else:
            if label != "Help":
                st.write(f"{icon} {label}: Not generated yet")

st.info("All files are saved in the `recordings/` folder on the server.")