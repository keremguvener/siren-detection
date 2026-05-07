"""
fft_analysis.py
---------------
Fourier donusumu ile frekans bolgesi analizi.
Sinyalin hangi frekanslari icerdigini buluyoruz.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_fft(signal, sr):
    """
    Sinyalin FFT'sini (Hizli Fourier Donusumu) hesaplar.
    Sadece pozitif frekanslari donduruyoruz cunku reel sinyallerde negatif kisim ayna goruntusudur.

    Parametreler:
        signal: numpy array, ses sinyali
        sr: int, sample rate

    Donus:
        freqs: frekans dizisi (Hz)
        mags: her frekansa karsilik gelen genlik (magnitude)
    """
    print("FFT hesaplaniyor...")

    N = len(signal)

    # numpy'in fft fonksiyonu karmasik sayilar dondurur, biz absolute alarak magnitude'a ceviriyoruz
    fft_sonucu = np.fft.fft(signal)
    mags = np.abs(fft_sonucu)

    # Frekans eksenini olustur: 0'dan Nyquist'e (sr/2) kadar
    freqs = np.fft.fftfreq(N, d=1.0 / sr)

    # Sadece pozitif frekans kismini al (ilk yarisi)
    yarim = N // 2
    freqs = freqs[:yarim]
    mags = mags[:yarim]

    # Genligi normalize edelim ki dosya boyutundan bagimsiz olsun
    mags = mags / N

    return freqs, mags


def plot_fft(freqs, mags, title, save_path):
    """
    FFT sonucunu (frekans spektrumunu) cizer.
    Sadece 0-4000 Hz arasini gosteriyoruz cunku sirenler bu bandda cikiyor.

    Parametreler:
        freqs: frekans dizisi
        mags: magnitude dizisi
        title: grafik basligi
        save_path: kayit yolu
    """
    print(f"FFT grafigi cizliyor: {title}")

    # 0-4000 Hz arasini filtrele (siren bandi burada)
    mask = freqs <= 4000
    freqs_kirpilmis = freqs[mask]
    mags_kirpilmis = mags[mask]

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_kirpilmis, mags_kirpilmis, color='darkorange', linewidth=0.8)
    plt.title(title)
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4000)
    plt.tight_layout()

    plt.savefig(save_path, dpi=100)
    plt.close()

    print(f"  -> Grafik kaydedildi: {save_path}")


def find_dominant_frequencies(freqs, mags, top_n=5):
    """
    En guclu N tane frekansi bulur.
    Bunlar sinyaldeki baskin frekans bilesenleridir.

    Parametreler:
        freqs: frekans dizisi
        mags: magnitude dizisi
        top_n: kac tane baskin frekans bulunsun

    Donus:
        baskin_frekanslar: liste, [(frekans, magnitude), ...] seklinde
    """
    # 0-4000 Hz arasini incele (siren bandinin disindakileri eleyelim)
    mask = (freqs >= 100) & (freqs <= 4000)
    f_kirpilmis = freqs[mask]
    m_kirpilmis = mags[mask]

    # En buyukten kucuge siralayan indexleri al, ilk N tanesini sec
    en_buyuk_indexler = np.argsort(m_kirpilmis)[::-1][:top_n]

    baskin_frekanslar = []
    for idx in en_buyuk_indexler:
        baskin_frekanslar.append((f_kirpilmis[idx], m_kirpilmis[idx]))

    return baskin_frekanslar
