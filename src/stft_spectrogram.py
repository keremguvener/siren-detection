"""
stft_spectrogram.py
-------------------
Kisa Sureli Fourier Donusumu (STFT) ve spektrogram modulu.
FFT tum sinyali tek seferde inceler, ama biz zamanla degisen frekansa bakmak istiyoruz.
Bu yuzden sinyali kucuk pencerelere bolup her birine ayri ayri FFT uyguluyoruz.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def compute_stft(signal, sr, window_size=1024, hop_length=512):
    """
    Sinyale STFT uygular.

    Parametreler:
        signal: ses sinyali
        sr: sample rate
        window_size: pencere boyutu (FFT noktasi)
        hop_length: pencereler arasi kayma miktari

    Donus:
        stft_matrix: karmasik degerli STFT matrisi (frekans x zaman)
        freqs: frekans dizisi
        times: zaman dizisi
    """
    print("STFT hesaplaniyor...")

    # librosa.stft pencereli FFT uygular, varsayilan pencere fonksiyonu Hann
    stft_matrix = librosa.stft(signal, n_fft=window_size, hop_length=hop_length)

    # Frekans ekseni (her satir bir frekans bandina karsilik geliyor)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_size)

    # Zaman ekseni (her sutun bir pencere zamanina karsilik geliyor)
    cerceve_sayisi = stft_matrix.shape[1]
    times = librosa.frames_to_time(np.arange(cerceve_sayisi), sr=sr, hop_length=hop_length)

    return stft_matrix, freqs, times


def plot_spectrogram(signal, sr, title, save_path):
    """
    Sinyalin spektrogramini cizer (zaman-frekans-genlik grafigi).
    dB olcegine ceviriyoruz cunku ses gucu logaritmik algilanir.

    Parametreler:
        signal: ses sinyali
        sr: sample rate
        title: grafik basligi
        save_path: kayit yolu
    """
    print(f"Spektrogram cizliyor: {title}")

    # STFT al ve magnitude'u dB'ye cevir
    stft_matrix = librosa.stft(signal, n_fft=1024, hop_length=512)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

    plt.figure(figsize=(10, 5))
    # specshow fonksiyonu spektrogram cizimi icin idealdir
    librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='hz',
        cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylim(0, 4000)  # siren bandina odaklan
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    print(f"  -> Spektrogram kaydedildi: {save_path}")


def track_dominant_frequency(stft_matrix, freqs):
    """
    Her zaman dilimi icin en yuksek enerjiye sahip frekansi bulur.
    Sirende bu frekans zamanla yukari-asagi oynar (siren karakteristigi).

    Parametreler:
        stft_matrix: STFT matrisi
        freqs: frekans dizisi

    Donus:
        dominant_freqs: numpy array, her zaman dilimi icin baskin frekans
    """
    # STFT matrisi karmasik, biz magnitude'a bakacagiz
    magnitude = np.abs(stft_matrix)

    # Sadece 0-4000 Hz arasini incele (yuksek frekanslarda gurultu olabilir)
    mask = freqs <= 4000
    magnitude_kirpilmis = magnitude[mask, :]
    freqs_kirpilmis = freqs[mask]

    # Her sutun (zaman dilimi) icin en buyuk magnitude'un bulundugu satiri bul
    en_yuksek_indexler = np.argmax(magnitude_kirpilmis, axis=0)

    # Bu indexlere karsilik gelen frekanslari al
    dominant_freqs = freqs_kirpilmis[en_yuksek_indexler]

    return dominant_freqs


def plot_dominant_frequency_track(dominant_freqs, times, title, save_path):
    """
    Zaman icinde baskin frekansin nasil degistigini cizer.
    Sirende bu egri salinim yapar, normal seste duz veya rastgele olur.

    Parametreler:
        dominant_freqs: zaman serisi olarak baskin frekanslar
        times: zaman dizisi
        title: grafik basligi
        save_path: kayit yolu
    """
    plt.figure(figsize=(10, 4))
    plt.plot(times, dominant_freqs, color='green', linewidth=1.0)
    plt.title(title)
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("Baskin Frekans (Hz)")
    plt.ylim(0, 4000)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    print(f"  -> Baskin frekans takibi kaydedildi: {save_path}")
