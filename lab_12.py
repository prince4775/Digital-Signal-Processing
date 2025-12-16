# dsp_lab_streamlit.py

import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from io import BytesIO
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------
# Streamlit Page Setup
# ----------------------
st.set_page_config(page_title="DSP Audio Lab: Averaging Filter Denoising", layout="wide")

# ---- Header ----
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("ayub.png", width=200)
with col_text:
    st.markdown("""
    ### **Muhammad Ayub — Reg No: 22jzele0470**
    # DSP Audio Processing Lab
    ### Record or Upload → Add Noise → **Averaging Filter Denoising**
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

# Ensure recordings folder
save_dir = ensure_folder("recordings")

# ----------------------
# Input Method Selection
# ----------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input:", ["Record Live", "Upload Audio File"])

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

                st.success("Recorded successfully!")
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
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.read())
            tfile_path = tfile.name
            tfile.close()

            try:
                audio_clean, fs = sf.read(tfile_path)
                if audio_clean.ndim > 1:
                    audio_clean = np.mean(audio_clean, axis=1)

                if audio_clean.dtype != np.float32:
                    audio_clean = audio_clean.astype(np.float32)
                    max_val = np.max(np.abs(audio_clean))
                    if max_val > 0:
                        audio_clean /= max_val

                ts = int(time.time())
                audio_clean_path = os.path.join(save_dir, f"clean_uploaded_{ts}.wav")
                save_wav(audio_clean_path, audio_clean, fs)

                st.session_state.audio_clean = audio_clean
                st.session_state.audio_clean_path = audio_clean_path
                st.session_state.fs = fs

                st.success(f"Uploaded: {uploaded_file.name}")
                st.audio(audio_bytes_from_array(audio_clean, fs).read(), format="audio/wav")

            except Exception as e:
                st.error(f"Error loading audio: {e}")
            finally:
                os.unlink(tfile_path)

# ======================
# Main Processing Flow
# ======================
if "audio_clean" in st.session_state:
    clean = st.session_state.audio_clean
    fs = st.session_state.get("fs", 44100)

    st.markdown("---")
    st.subheader("Step 1: Clean Audio Analysis")

    tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])
    with tab1:
        st.pyplot(plot_waveform(clean, "Clean Signal - Time Domain", 'green'))
    with tab2:
        st.pyplot(plot_spectrum(clean, fs, "Clean Signal - Frequency Domain"))
    with tab3:
        st.pyplot(plot_spectrogram(clean, fs, "Clean Signal - Spectrogram"))

    # ----------------------
    # Add Noise Section
    # ----------------------
    st.markdown("---")
    st.subheader("Step 2: Add Noise")

    colA, colB = st.columns(2)
    with colA:
        noise_type = st.selectbox("Noise Type", ["White Gaussian", "Uniform"], index=0)
        noise_level = st.slider("Manual Noise STD", 0.001, 0.3, 0.05, step=0.001)
    with colB:
        target_snr = st.slider("Target SNR (dB)", -15, 40, 10)
        use_snr = st.checkbox("Use Target SNR", value=True)

    if st.button("Generate Noisy Audio", type="secondary"):
        with st.spinner("Adding noise..."):
            if use_snr:
                signal_power = np.mean(clean ** 2)
                noise_power = signal_power / (10 ** (target_snr / 10.0))
                noise_std = np.sqrt(noise_power)
            else:
                noise_std = noise_level

            if noise_type == "White Gaussian":
                noise = np.random.normal(0, noise_std, len(clean))
            else:  # Uniform
                noise = np.random.uniform(-noise_std * np.sqrt(3), noise_std * np.sqrt(3), len(clean))

            noisy = clean + noise

            ts = int(time.time())
            noisy_path = os.path.join(save_dir, f"noisy_{ts}.wav")
            save_wav(noisy_path, noisy, fs)

            st.session_state.audio_noisy = noisy
            st.session_state.audio_noisy_path = noisy_path

            st.success("Noisy audio created!")
            st.audio(audio_bytes_from_array(noisy, fs).read(), format="audio/wav")

    # ----------------------
    # Noisy Audio Analysis
    # ----------------------
    if "audio_noisy" in st.session_state:
        noisy = st.session_state.audio_noisy

        st.markdown("---")
        st.subheader("Step 2: Noisy Audio Analysis")

        tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])
        with tab1:
            st.pyplot(plot_waveform(noisy, "Noisy Signal - Time Domain", 'red'))
        with tab2:
            st.pyplot(plot_spectrum(noisy, fs, "Noisy Signal - Frequency Domain"))
        with tab3:
            st.pyplot(plot_spectrogram(noisy, fs, "Noisy Signal - Spectrogram"))

        # ----------------------
        # Averaging Filter Section
        # ----------------------
        st.markdown("---")
        st.subheader("Step 3: Averaging (Moving Average) Filter Denoising")

        filter_length = st.slider(
            "Averaging Filter Length (N points)", 
            min_value=3, max_value=99, value=21, step=2
        )  # Odd numbers only

        if st.button("Apply Averaging Filter", type="primary"):
            with st.spinner(f"Applying {filter_length}-point averaging filter..."):
                kernel = np.ones(filter_length) / filter_length
                denoised = np.convolve(noisy, kernel, mode='same')

                max_val = np.max(np.abs(denoised))
                if max_val > 0:
                    denoised = denoised / max_val * 0.95

                ts = int(time.time())
                denoised_path = os.path.join(save_dir, f"denoised_averaging_{filter_length}pt_{ts}.wav")
                save_wav(denoised_path, denoised, fs)

                st.session_state.audio_denoised = denoised
                st.session_state.audio_denoised_path = denoised_path

                st.success(f"Averaging filter applied (N={filter_length})!")
                st.audio(audio_bytes_from_array(denoised, fs).read(), format="audio/wav")

        # ----------------------
        # Denoised Audio Analysis
        # ----------------------
        if "audio_denoised" in st.session_state:
            denoised = st.session_state.audio_denoised

            st.markdown("---")
            st.subheader("Step 3: Denoised Audio Analysis")

            tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Spectrogram"])
            with tab1:
                st.pyplot(plot_waveform(denoised, "Denoised Signal - Time Domain", 'blue'))
            with tab2:
                st.pyplot(plot_spectrum(denoised, fs, "Denoised Signal - Frequency Domain"))
            with tab3:
                st.pyplot(plot_spectrogram(denoised, fs, "Denoised Signal - Spectrogram"))

            # ----------------------
            # Final Comparison
            # ----------------------
            st.markdown("---")
            st.subheader("Final Comparison: Clean → Noisy → Denoised")

            comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Time Domain Comparison", "Frequency Domain Comparison", "Spectrogram Comparison"])

            with comp_tab1:
                c1, c2, c3 = st.columns(3)
                with c1: st.pyplot(plot_waveform(clean, "Clean", 'green'))
                with c2: st.pyplot(plot_waveform(noisy, "Noisy", 'red'))
                with c3: st.pyplot(plot_waveform(denoised, f"Denoised (N={filter_length})", 'blue'))

            with comp_tab2:
                c1, c2, c3 = st.columns(3)
                with c1: st.pyplot(plot_spectrum(clean, fs, "Clean Spectrum"))
                with c2: st.pyplot(plot_spectrum(noisy, fs, "Noisy Spectrum"))
                with c3: st.pyplot(plot_spectrum(denoised, fs, "Denoised Spectrum"))

            with comp_tab3:
                c1, c2, c3 = st.columns(3)
                with c1: st.pyplot(plot_spectrogram(clean, fs, "Clean"))
                with c2: st.pyplot(plot_spectrogram(noisy, fs, "Noisy"))
                with c3: st.pyplot(plot_spectrogram(denoised, fs, "Denoised"))

            # ----------------------
            # Performance Metrics Section (Using scikit-learn)
            # ----------------------
            st.markdown("---")
            st.subheader("📊 Performance Metrics (Denoising Evaluation)")

            # Trim to same length
            min_len = min(len(clean), len(noisy), len(denoised))
            clean_trim = clean[:min_len]
            noisy_trim = noisy[:min_len]
            denoised_trim = denoised[:min_len]

            # SNR function
            def calculate_snr(reference, processed):
                signal_power = np.mean(reference ** 2)
                noise_power = np.mean((reference - processed) ** 2)
                if noise_power == 0:
                    return float('inf')
                return 10 * np.log10(signal_power / noise_power)

            snr_noisy = calculate_snr(clean_trim, noisy_trim)
            snr_denoised = calculate_snr(clean_trim, denoised_trim)
            snr_improvement = snr_denoised - snr_noisy

            mse_noisy = mean_squared_error(clean_trim, noisy_trim)
            mse_denoised = mean_squared_error(clean_trim, denoised_trim)

            mae_denoised = mean_absolute_error(clean_trim, denoised_trim)

            peak = 1.0
            psnr_noisy = 10 * np.log10(peak**2 / mse_noisy) if mse_noisy > 0 else float('inf')
            psnr_denoised = 10 * np.log10(peak**2 / mse_denoised) if mse_denoised > 0 else float('inf')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Signal-to-Noise Ratio (SNR)**")
                st.metric("Noisy Audio", f"{snr_noisy:.2f} dB")
                st.metric("Denoised Audio", f"{snr_denoised:.2f} dB")
                st.metric("**SNR Improvement**", f"+{snr_improvement:.2f} dB", delta=f"{snr_improvement:.2f} dB")

            with col2:
                st.markdown("**Error Metrics**")
                st.metric("MSE (Noisy)", f"{mse_noisy:.6f}")
                st.metric("MSE (Denoised)", f"{mse_denoised:.6f}", delta=f"-{mse_noisy - mse_denoised:.6f}")
                st.metric("MAE (Denoised)", f"{mae_denoised:.6f}")

            with col3:
                st.markdown("**Peak Signal-to-Noise Ratio (PSNR)**")
                st.metric("Noisy Audio", f"{psnr_noisy:.2f} dB")
                st.metric("Denoised Audio", f"{psnr_denoised:.2f} dB")
                st.metric("PSNR Gain", f"+{psnr_denoised - psnr_noisy:.2f} dB")

            st.success(f"""
            **Summary**:  
            The averaging filter improved SNR by **{snr_improvement:.2f} dB**.  
            • Higher SNR & PSNR = Better quality  
            • Lower MSE & MAE = Closer to original clean signal  
            Experiment with different filter lengths to optimize performance!
            """)

# ======================
# Download Section
# ======================
st.markdown("---")
st.subheader("Download Processed Files")

cols = st.columns(3)
paths = [
    st.session_state.get("audio_clean_path"),
    st.session_state.get("audio_noisy_path"),
    st.session_state.get("audio_denoised_path")
]
labels = ["Clean Audio", "Noisy Audio", "Denoised (Averaging Filter)"]
icons = ["🟢", "🔴", "🔵"]

for col, path, label, icon in zip(cols, paths, labels, icons):
    with col:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    label=f"{icon} Download {label}",
                    data=f,
                    file_name=os.path.basename(path),
                    mime="audio/wav"
                )
        else:
            st.write(f"{icon} {label}: Not ready yet")

st.info("All files are saved in the `recordings/` folder on the server.")