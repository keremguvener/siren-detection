"""
filtering.py
------------
Bandpass filtre uygulama modulu.
Sirenler genelde 500-1800 Hz arasinda calar, biz de bu bandi gecirmek icin Butterworth filtresi kuruyoruz.
Boylece gurultuyu ve siren disi sesleri zayiflatmis oluyoruz.
"""

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def bandpass_filter(signal, sr, lowcut=500, highcut=1800, order=5):
    """
    Sinyale Butterworth bandpass filtre uygular.
    Sadece [lowcut, highcut] Hz arasindaki frekanslari geciriyoruz.

    Parametreler:
        signal: ses sinyali (numpy array)
        sr: sample rate
        lowcut: alt kesim frekansi (Hz), default 500
        highcut: ust kesim frekansi (Hz), default 1800
        order: filtre derecesi, default 5

    Donus:
        filtered_signal: filtreden gecmis sinyal
    """
    print(f"Bandpass filtre uygulaniyor ({lowcut}-{highcut} Hz)...")

    # Nyquist frekansi (sample rate'in yarisi)
    nyquist = 0.5 * sr

    # Kesim frekanslarini Nyquist'e gore normalize et (0-1 arasi)
    low = lowcut / nyquist
    high = highcut / nyquist

    # Butterworth filtre katsayilarini olustur
    # b: pay katsayilari, a: payda katsayilari
    b, a = butter(order, [low, high], btype='band')

    # filtfilt sifir-faz filtreleme yapar (sinyali ileri+geri filtreliyor)
    # boylece faz kaymasi olmuyor, ama filtre cevabi karelenmis gibi olur
    filtered_signal = filtfilt(b, a, signal)

    # Filtreleme oncesi/sonrasi enerji karsilastirmasi
    enerji_oncesi = np.sum(signal ** 2)
    enerji_sonrasi = np.sum(filtered_signal ** 2)
    print(f"  -> Enerji oncesi: {enerji_oncesi:.2f}")
    print(f"  -> Enerji sonrasi: {enerji_sonrasi:.2f}")
    print(f"  -> Filtre uygulandi.")

    return filtered_signal


def plot_filtered_comparison(original, filtered, sr, title, save_path):
    """
    Orijinal ve filtrelenmis sinyalleri yan yana cizer.
    Filtrenin etkisini gormek icin gorsel karsilastirma sagliyor.

    Parametreler:
        original: filtreleme oncesi sinyal
        filtered: filtreleme sonrasi sinyal
        sr: sample rate
        title: grafik basligi
        save_path: kayit yolu
    """
    sure = len(original) / sr
    zaman = np.linspace(0, sure, len(original))

    # Iki alt grafik olustur (alt alta)
    fig, eksenler = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    eksenler[0].plot(zaman, original, color='steelblue', linewidth=0.6)
    eksenler[0].set_title("Orijinal Sinyal")
    eksenler[0].set_ylabel("Genlik")
    eksenler[0].grid(True, alpha=0.3)

    eksenler[1].plot(zaman, filtered, color='crimson', linewidth=0.6)
    eksenler[1].set_title("Bandpass Filtrelenmis Sinyal (500-1800 Hz)")
    eksenler[1].set_xlabel("Zaman (saniye)")
    eksenler[1].set_ylabel("Genlik")
    eksenler[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    print(f"  -> Karsilastirma grafigi kaydedildi: {save_path}")
