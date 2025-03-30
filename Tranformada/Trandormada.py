import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(signal, sampling_rate, plot=False):
    """
    Calcula la Transformada de Fourier de una señal y opcionalmente la grafica.
    
    Parámetros:
    signal (array): Señal de entrada en el dominio del tiempo.
    sampling_rate (float): Frecuencia de muestreo de la señal.
    plot (bool): Si es True, grafica la magnitud del espectro de la señal.
    
    Retorna:
    freqs (array): Frecuencias correspondientes a la Transformada de Fourier.
    spectrum (array): Magnitud de la Transformada de Fourier.
    """
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/sampling_rate)
    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))  # Graficamos solo la parte positiva
        plt.title("Transformada de Fourier")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.grid()
        plt.show()
    
    return freqs, np.abs(spectrum)


if __name__ == "__main__":
    sampling_rate = 1000  # Hz
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    
    fourier_transform(signal, sampling_rate, plot=True)