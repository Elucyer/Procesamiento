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


def pan_tompkins(signal, fs):
    # Tiempo original
    t = np.linspace(0, len(signal) / fs, len(signal))

    # 1. Filtro pasa banda (5–15 Hz)
    def bandpass_filter(sig, lowcut=5.0, highcut=15.0, fs=360, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, sig)

    filtered = bandpass_filter(signal, fs=fs)

    # 2. Derivada
    diff = np.diff(filtered)
    t_diff = t[1:]  # por la derivada

    # 3. Cuadrado
    squared = diff ** 2
    t_squared = t_diff

    # 4. Integración (ventana de 150 ms)
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    t_integrated = t[:len(integrated)]  # ajustar al largo de la señal

    # 5. Detección de picos
    threshold = np.mean(integrated) * 1.5
    peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.2 * fs))

    # --------- GRAFICAR LOS PASOS ---------
    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(t, signal)
    plt.title("ECG Original")

    plt.subplot(5, 1, 2)
    plt.plot(t, filtered)
    plt.title("ECG Filtrado (5–15 Hz)")

    plt.subplot(5, 1, 3)
    plt.plot(t_diff, diff)
    plt.title("Derivada")

    plt.subplot(5, 1, 4)
    plt.plot(t_squared, squared)
    plt.title("Señal Cuadrada")

    plt.subplot(5, 1, 5)
    plt.plot(t_integrated, integrated)
    plt.plot(t_integrated[peaks], integrated[peaks], 'ro')
    plt.title("Integración y Picos R detectados")
    plt.xlabel("Tiempo (s)")

    plt.tight_layout()
    plt.show()

    return peaks


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


def calcular_frecuencia_cardíaca_pantompkin(peaks, fs):
    """
    Calcula la frecuencia cardíaca en BPM a partir de los índices de los picos R.

    Parámetros:
    - peaks: array de índices de los picos R
    - fs: frecuencia de muestreo (Hz)

    Retorna:
    - bpm: frecuencia cardíaca estimada (latidos por minuto)
    - cantidad de picos detectados
    """

    # Si hay menos de 2 picos, no se puede calcular diferencia
    if len(peaks) < 2:
        return 0, 0

    # Duración total en segundos entre el primer y último pico
    duracion_seg = (peaks[-1] - peaks[0]) / fs
    cantidad_latidos = len(peaks)

    # BPM = latidos / minutos
    bpm = (cantidad_latidos / duracion_seg) * 60

    return bpm, cantidad_latidos