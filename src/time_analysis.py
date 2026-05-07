"""
time_analysis.py
----------------
Zaman bolgesi analizi icin fonksiyonlar.
Sinyali zaman ekseninde gorsellestiriyor, RMS enerjisini hesapliyoruz.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain(signal, sr, title, save_path):
    """
    Sinyalin zaman bolgesi grafigini cizer (genlik vs zaman).
    Ayrica RMS (Root Mean Square) enerjisini hesaplar ve dondurur.

    Parametreler:
        signal: numpy array, ses sinyali
        sr: int, sample rate
        title: string, grafik basligi
        save_path: string, grafigin kaydedilecegi yol

    Donus:
        rms: float, sinyalin RMS enerji degeri
    """
    print(f"Zaman bolgesi grafigi cizliyor: {title}")

    # Zaman eksenini olustur: 0'dan sinyal suresine kadar, ornek sayisi kadar nokta
    sure = len(signal) / sr
    zaman = np.linspace(0, sure, len(signal))

    # RMS enerjisi hesabi: ortalama karelerinin karekoku
    # Sinyal ne kadar guclu, ne kadar enerjili ona bakiyoruz
    rms = np.sqrt(np.mean(signal ** 2))

    # Grafigi cizdiriyoruz
    plt.figure(figsize=(10, 4))
    plt.plot(zaman, signal, color='steelblue', linewidth=0.6)
    plt.title(title + f"  (RMS = {rms:.4f})")
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("Genlik")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Dosyaya kaydet ve figuru kapat (bellek sismesin diye)
    plt.savefig(save_path, dpi=100)
    plt.close()

    print(f"  -> RMS enerjisi: {rms:.4f}")
    print(f"  -> Grafik kaydedildi: {save_path}")

    return rms
