import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, freqz, tf2zpk, spectrogram, find_peaks
from scipy.fftpack import fft



def cargar_senal(filepath, tipo='mat', variable='emg', fs=None):
    """
    Carga una señal desde un archivo .mat, .csv, .wav, .mp3, .mp4
    """
    if tipo == 'mat':
        data = loadmat(filepath)
        signal = data[variable][:, 0]
        if fs is None:
            fs = 1000  # Asume 1000 Hz por defecto
    elif tipo == 'csv':
        signal = np.loadtxt(filepath, delimiter=',')
        if fs is None:
            fs = 360  # ECG común
    elif tipo in ['wav', 'mp3', 'mp4']:
        signal, fs = librosa.load(filepath, sr=None, mono=True)
    else:
        raise ValueError("Tipo no soportado. Usa 'mat', 'csv', 'wav', 'mp3', o 'mp4'.")

    t = np.arange(len(signal)) / fs
    return signal, t, fs


def filtrar_senal(signal, fs, lowcut, highcut, order=4):
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered, b, a


def graficar_filtro(b, a, fs):
    w, h = freqz(b, a, worN=8000)
    plt.figure(figsize=(12, 4))
    plt.plot(w * fs / (2 * np.pi), 20 * np.log10(abs(h)))
    plt.title("Respuesta en Frecuencia del Filtro")
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Ganancia (dB)')
    plt.grid(True)
    plt.show()

    z, p, _ = tf2zpk(b, a)
    plt.figure()
    plt.title('Plano Z del Filtro')
    plt.scatter(np.real(z), np.imag(z), marker='o', label='Ceros')
    plt.scatter(np.real(p), np.imag(p), marker='x', label='Polos')
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    plt.gca().add_artist(unit_circle)
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def teager_kaiser(signal):
    return signal[1:-1]**2 - signal[:-2]*signal[2:]

def calcular_rms(signal, window_size):
    return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid'))

def detectar_onset_offset(rms_signal, t, threshold_ratio=0.4):
    threshold = np.max(rms_signal) * threshold_ratio
    active = rms_signal > threshold
    onsets = np.where(np.diff(active.astype(int)) == 1)[0]
    offsets = np.where(np.diff(active.astype(int)) == -1)[0]
    return onsets, offsets, threshold


def graficar_senal(t, signal, titulo="Señal", xlabel="Tiempo (s)", ylabel="Amplitud"):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def graficar_onset_offset(t, rms_signal, onsets, offsets):
    plt.figure(figsize=(12, 4))
    plt.plot(t[:len(rms_signal)], rms_signal, label='RMS')
    for o in onsets:
        plt.axvline(t[o], color='g', linestyle='--', label='Onset')
    for o in offsets:
        plt.axvline(t[o], color='r', linestyle='--', label='Offset')
    plt.title("Onset y Offset sobre RMS")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud RMS")
    plt.legend()
    plt.grid(True)
    plt.show()


def fft_y_espectrograma(signal, fs):
    N = len(signal)
    f = np.fft.fftfreq(N, 1/fs)
    fft_result = np.abs(fft(signal))

    # Calcular espectrograma
    f_spec, t_spec, Sxx = spectrogram(signal, fs, nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Convertir a dB

    # Graficar FFT
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f[:N//2], fft_result[:N//2])
    plt.title("Transformada de Fourier")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.xlim([0, 300])

    # Graficar espectrograma
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_spec, f_spec, Sxx_db, shading='auto', cmap='plasma')
    plt.title("Espectrograma (dB)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.colorbar(label='dB')  # Escala de colores

    # Opcional: limitar frecuencia para enfoque en región audible
    plt.ylim([0, 2000])

    plt.tight_layout()
    plt.show()


def detectar_picos_r(ecg_signal, fs):
    ecg_diff = np.diff(ecg_signal)
    ecg_sq = ecg_diff ** 2
    ecg_integrated = np.convolve(ecg_sq, np.ones(30)/30, mode='same')

    peaks, _ = find_peaks(ecg_integrated, distance=fs*0.3, height=np.max(ecg_integrated)*0.5)

    plt.figure(figsize=(10, 6))
    plt.plot(ecg_integrated)
    plt.plot(peaks, ecg_integrated[peaks], 'rx')
    plt.title("Detección de picos R")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()

    return peaks

def calcular_frecuencia_cardiaca(peaks, fs):
    tiempos_r = peaks / fs
    rr_intervals = np.diff(tiempos_r)
    hr_bpm = 38 / np.mean(rr_intervals)
    return hr_bpm