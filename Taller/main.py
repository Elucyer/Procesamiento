import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.signal import butter, filtfilt, freqz, tf2zpk, spectrogram, find_peaks
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft
import soundfile as sf
import numpy as np
import threading
import librosa



def cargar_senal(filepath, tipo='mat', variable='emg', fs=None):
    """
    Carga una señal desde un archivo .mat, .csv, .wav, .mp3, .mp4
    y devuelve la señal, el vector de tiempo y la frecuencia de muestreo.
    """
    if tipo == 'mat':
        data = loadmat(filepath)

        if variable not in data:
            posibles = [k for k in data.keys() if not k.startswith('__')]
            if len(posibles) == 1:
                variable = posibles[0]
            else:
                raise KeyError(f"Variable '{variable}' no encontrada. Variables disponibles: {posibles}")
        signal = np.squeeze(data[variable])
        if isinstance(signal[0], np.ndarray) and signal[0].ndim == 2:
            signal = np.concatenate([x.flatten() for x in signal])
        if fs is None:
            fs = 1000
    elif tipo == 'csv':
        signal = np.loadtxt(filepath, delimiter=',')
        if fs is None:
            fs = 360
    elif tipo in ['wav', 'mp3', 'mp4']:
        signal, fs = librosa.load(filepath, sr=None, mono=True)
    else:
        raise ValueError("Tipo no soportado. Usa 'mat', 'csv', 'wav', 'mp3', o 'mp4'.")

    t = np.arange(len(signal)) / fs
    return signal, t, fs


def filtrar_senal(signal, fs, lowcut, highcut, order=4):
    """
    -- Se define y crea el filtro segun la necesidad de la senal
    """
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered, b, a


def graficar_filtro(b, a, fs):
    """
    -- Funcion para graficar filtros
    """
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

def detectar_onset_offset_auto(rms_signal, t, min_ratio=0.05, max_ratio=0.4, step=0.05, verbose=True):
    """
    Detecta onsets y offsets ajustando automáticamente el threshold.
    Asegura que cada onset tenga un offset posterior.

    Retorna:
    - onsets: índices de inicio
    - offsets: índices de fin
    - threshold_ratio usado
    - threshold absoluto
    """
    for ratio in np.arange(max_ratio, min_ratio - step, -step):
        threshold = np.max(rms_signal) * ratio
        active = rms_signal > threshold
        onsets = np.where(np.diff(active.astype(int)) == 1)[0]
        offsets = np.where(np.diff(active.astype(int)) == -1)[0]

        # Asegurar que los pares estén alineados
        if len(onsets) > 0 and len(offsets) > 0:
            # Caso 1: primer offset ocurre antes del primer onset → descartarlo
            if offsets[0] < onsets[0]:
                offsets = offsets[1:]

            # Caso 2: más onsets que offsets → quitar el último onset
            if len(onsets) > len(offsets):
                onsets = onsets[:len(offsets)]

            # Caso 3: más offsets que onsets → quitar el último offset
            elif len(offsets) > len(onsets):
                offsets = offsets[:len(onsets)]

            if len(onsets) > 0:
                if verbose:
                    print(f"Detección con threshold_ratio = {ratio:.2f}")
                    print(f"Pares válidos: {len(onsets)} (onsets y offsets)")
                return onsets, offsets, ratio, threshold

    if verbose:
        print("⚠️ No se detectaron pares válidos de onset/offset.")
    return [], [], None, None


def graficar_senal(t, signal, titulo="Señal", xlabel="Tiempo (s)", ylabel="Amplitud"):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def graficar_onset_offset(t, onsets, offsets):
    plt.figure(figsize=(12, 4))
    first_onset = True
    for o in onsets:
        label = 'Onset' if first_onset else ""
        plt.axvline(t[o], color='g', linestyle='--', label=label)
        first_onset = False

    first_offset = True
    for o in offsets:
        label = 'Offset' if first_offset else ""
        plt.axvline(t[o], color='r', linestyle='--', label=label)
        first_offset = False

    plt.title("Onset y Offset")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
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
    t_diff = t[1:]

    # 3. Cuadrado
    squared = diff ** 2
    t_squared = t_diff

    # 4. Integración
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


def graficar_resultados_emg(t, emg_original, emg_filtrada, tk_signal, rms, t_rms, onsets, offsets):
    """
    Genera una figura con subplots mostrando:
    - EMG original
    - EMG filtrada con Onset/Offset + regiones sombreadas
    - Teager-Kaiser
    - RMS
    """
    tk_time = t[1:-1]  # Ajustar tiempo para Teager-Kaiser

    plt.figure(figsize=(12, 10))

    # 1. Señal original
    plt.subplot(4, 1, 1)
    plt.plot(t, emg_original)
    plt.title("EMG Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)

    # 2. EMG Filtrada + Onset/Offset + Regiones activas
    plt.subplot(4, 1, 2)
    plt.plot(t, emg_filtrada, label='EMG Filtrada')

    # Dibujar líneas y sombrear regiones entre onsets y offsets
    first_onset = True
    first_offset = True
    for i in range(min(len(onsets), len(offsets))):
        o_start = onsets[i]
        o_end = offsets[i]
        if o_start < len(t_rms) and o_end < len(t_rms):
            plt.axvline(t_rms[o_start], color='green', linestyle='--', label='Onset' if first_onset else "")
            plt.axvline(t_rms[o_end], color='red', linestyle='--', label='Offset' if first_offset else "")
            plt.axvspan(t_rms[o_start], t_rms[o_end], color='orange', alpha=0.2)
            first_onset = False
            first_offset = False

    plt.title("EMG Filtrada (20–450 Hz) + Onset/Offset + Contracciones")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()

    # 3. Teager-Kaiser
    plt.subplot(4, 1, 3)
    plt.plot(tk_time, tk_signal)
    plt.title("Operador Teager-Kaiser")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía")
    plt.grid(True)

    # 4. RMS
    plt.subplot(4, 1, 4)
    plt.plot(t_rms, rms, label="RMS")
    plt.title("RMS de la Señal EMG (post-TK)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("RMS")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def recortar_segmento(t, signal, t_min, t_max):
    indices = np.where((t >= t_min) & (t <= t_max))[0]
    return signal[indices], t[indices]


def graficar_teager_y_rms(t_tk, tk_signal, t_rms, rms):
    """
    Gráfico con 2 subplots:
    - Teager-Kaiser (energía de la señal EMG)
    - RMS de la señal TK con ventana deslizante
    """
    plt.figure(figsize=(12, 6))

    # Subplot 1: Teager-Kaiser
    plt.subplot(2, 1, 1)
    plt.plot(t_tk, tk_signal, label='Teager-Kaiser', color='blue')
    plt.title("Señal EMG - Operador Teager-Kaiser")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía")
    plt.grid(True)
    plt.legend()

    # Subplot 2: RMS del TK
    plt.subplot(2, 1, 2)
    plt.plot(t_rms, rms, label='RMS', color='red')
    plt.title("RMS de la Señal Teager-Kaiser con Ventana Deslizante")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Valor RMS")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def graficar_zoom_contracciones(t_rms, rms, onsets, offsets, zoom_duracion=2):
    """
    Genera subplots con zoom a cada par onset-offset usando RMS.
    - zoom_duracion: tiempo total (en segundos) a mostrar alrededor del evento
    """
    plt.figure(figsize=(12, len(onsets) * 2))

    for i in range(len(onsets)):
        if i >= len(offsets):
            break

        onset_time = t_rms[onsets[i]]
        offset_time = t_rms[offsets[i]]

        # Definir ventana de zoom (centrada en el evento)
        zoom_center = (onset_time + offset_time) / 2
        zoom_range = zoom_duracion / 2
        t_min = zoom_center - zoom_range
        t_max = zoom_center + zoom_range

        # Sombra para el rango
        mask = (t_rms >= t_min) & (t_rms <= t_max)

        plt.subplot(len(onsets), 1, i + 1)
        plt.plot(t_rms[mask], rms[mask], label="RMS", color='blue')
        plt.axvline(onset_time, color='green', linestyle='--', label='Onset')
        plt.axvline(offset_time, color='red', linestyle='--', label='Offset')
        plt.axvspan(onset_time, offset_time, color='orange', alpha=0.3)

        plt.title(f"Zoom Contracción {i+1}")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("RMS")
        plt.grid(True)
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()

def cargar_audios_desde_carpeta(carpeta):
    """
    -- Esta funcion sirve para listar los elementos de una carpeta en este caso audios
    -- en diferentes formatos
    """
    extensiones_validas = ('.wav', '.flac', '.aiff', '.aif', '.mp3', '.mp4')
    audio_files = [f for f in os.listdir(carpeta) if f.lower().endswith(extensiones_validas)]

    audios = {}
    for filename in audio_files:
        path = os.path.join(carpeta, filename)
        try:
            if filename.lower().endswith(('.mp3', '.mp4')):
                data, samplerate = librosa.load(path, sr=None, mono=True)
            else:
                data, samplerate = sf.read(path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
            audios[filename] = {'data': data, 'samplerate': samplerate}
            print(f"Cargado: {filename}")
        except Exception as e:
            print(f"Error cargando {filename}: {e}")

    return audios

def escuchar_y_animar_audio(nombre, data, samplerate):
    from IPython.display import Audio, display

    print(f"\n▶️ Reproduciendo: {nombre}")
    duracion = len(data) / samplerate
    t = np.linspace(0, duracion, num=len(data))

    def reproducir_audio():
        if 'google.colab' in sys.modules:
            display(Audio(data, rate=samplerate))
        else:
            try:
                import sounddevice as sd
                sd.play(data, samplerate)
                sd.wait()
            except ImportError:
                print("❌ sounddevice no está instalado. Usa `pip install sounddevice`")

    # Si NO estás en Colab → lanzar hilo para reproducir en paralelo
    if 'google.colab' not in sys.modules:
        hilo_audio = threading.Thread(target=reproducir_audio)
        hilo_audio.start()
    else:
        reproducir_audio()

    # Gráfico y muestra el audio en pantalla
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, data, label='Audio')
    barra = ax.axvline(0, color='red', linestyle='--', label='Posición actual')  # barra animada
    ax.set_xlim(0, duracion)
    ax.set_ylim(np.min(data) * 1.1, np.max(data) * 1.1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title(f"Reproduciendo: {nombre}")
    ax.grid(True)
    ax.legend()

    # Actualiza una barra de progreso que corresponde al tiempo del audio en la grafica
    def actualizar(frame):
        tiempo_actual = frame / fps
        barra.set_xdata([tiempo_actual, tiempo_actual])
        return barra,

    fps = 30
    total_frames = int(duracion * fps)

    anim = FuncAnimation(fig, actualizar, frames=total_frames, interval=1000/fps, blit=True)

    plt.tight_layout()
    plt.show()

    # Esperar finalización del hilo si se usó
    if 'google.colab' not in sys.modules:
        hilo_audio.join()