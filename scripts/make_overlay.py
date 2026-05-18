"""
make_overlay.py
---------------
Daha zor bir test: bir siren ve bir no_siren WAV'ini ayni anda calar gibi
ust uste bindirir (toplama).

Yontem:
  karisim = SIREN_GAIN * siren + BG_GAIN * no_siren
  -> sonra |karisim|.max() ile normalize edilir (clipping olmasin)

Bunu 3 farkli SNR ayariyla deneyelim:
  - egitim: siren ve gurultu esit (1.0, 1.0)
  - orta : siren biraz yuksek (1.0, 0.7)
  - zor  : gurultu siren'den yuksek (0.6, 1.0) - bu en zorlu case

Her uretilen dosya icin pipeline + detect_siren calistirilir.

Kullanim:
    python scripts/make_overlay.py
"""

import glob
import os
import sys

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJE_KOKU)

import numpy as np
import soundfile as sf

from src.load_audio import load_wav
from src.time_analysis import plot_time_domain
from src.fft_analysis import compute_fft, plot_fft, find_dominant_frequencies
from src.filtering import bandpass_filter, plot_filtered_comparison
from src.stft_spectrogram import (
    compute_stft,
    plot_spectrogram,
    track_dominant_frequency,
    plot_dominant_frequency_track,
)
from src.detector import detect_siren


DATA_DIR = os.path.join(PROJE_KOKU, "data")
PLOTS_DIR = os.path.join(PROJE_KOKU, "plots")
SIREN_DIR = os.path.join(DATA_DIR, "siren")
NOSIREN_DIR = os.path.join(DATA_DIR, "no_siren")
CUSTOM_DIR = os.path.join(DATA_DIR, "custom")

TERCIH_SIREN = "1-31482-B-42"
TERCIH_NOSIREN = "1-30709-C-23"

# (siren_gain, bg_gain, cikti_ismi)
# Esit -> Orta -> Zor (gurultu daha yuksek)
SENARYOLAR = [
    (1.0, 1.0, "overlay_esit"),
    (1.0, 0.7, "overlay_orta"),
    (0.6, 1.0, "overlay_zor"),
]


def dosya_sec(dosyalar, anahtar):
    for p in dosyalar:
        if anahtar in p:
            return p
    return dosyalar[0]


def ustuste_bindir(siren_yolu, nosiren_yolu, siren_gain, bg_gain, cikti_yolu):
    """
    Iki sinyali toplama ile karistirir, normalize edip kaydeder.
    Sinyaller ayni uzunlukta (her ikisi de 5 saniye ESC-50). Yine de
    guvenli olsun diye min uzunluga kirpiyoruz.
    """
    print(f"\n[OVERLAY] siren_gain={siren_gain}, bg_gain={bg_gain}")
    print(f"  siren    : {os.path.basename(siren_yolu)}")
    print(f"  no_siren : {os.path.basename(nosiren_yolu)}")

    s, sr_s = load_wav(siren_yolu)
    b, sr_b = load_wav(nosiren_yolu)
    assert sr_s == sr_b, "Sample rate uyusmazligi"

    n = min(len(s), len(b))
    s = s[:n]
    b = b[:n]

    karisim = siren_gain * s + bg_gain * b

    # Clipping'i onlemek icin tepe degerini 1'in altinda tutalim
    tepe = np.max(np.abs(karisim))
    if tepe > 0:
        karisim = karisim / tepe * 0.99

    sf.write(cikti_yolu, karisim, sr_s)
    print(f"  kaydedildi: {cikti_yolu}  ({n/sr_s:.2f} s)")
    return sr_s


def pipeline_calistir(wav_yolu, baslik, plots_klasoru):
    """main.py'deki dosya_isle ile ayni adimlar."""
    os.makedirs(plots_klasoru, exist_ok=True)
    signal, sr = load_wav(wav_yolu)

    plot_time_domain(signal, sr, f"{baslik} - Zaman Bolgesi",
                     os.path.join(plots_klasoru, "01_zaman.png"))
    freqs, mags = compute_fft(signal, sr)
    plot_fft(freqs, mags, f"{baslik} - FFT",
             os.path.join(plots_klasoru, "02_fft.png"))
    baskin = find_dominant_frequencies(freqs, mags, top_n=5)
    print("  Baskin frekanslar (ilk 5):")
    for f, m in baskin:
        print(f"     {f:.1f} Hz  (mag={m:.5f})")

    filtered = bandpass_filter(signal, sr, lowcut=500, highcut=1800, order=5)
    plot_filtered_comparison(signal, filtered, sr,
                             f"{baslik} - Filtre",
                             os.path.join(plots_klasoru, "03_filtre.png"))
    plot_spectrogram(signal, sr, f"{baslik} - Spektrogram",
                     os.path.join(plots_klasoru, "04_spektrogram.png"))
    stft_matrix, ff, tt = compute_stft(filtered, sr)
    dom = track_dominant_frequency(stft_matrix, ff)
    plot_dominant_frequency_track(dom, tt, f"{baslik} - Baskin Frekans",
                                  os.path.join(plots_klasoru, "05_baskin_frekans.png"))

    return detect_siren(wav_yolu)


def main():
    os.makedirs(CUSTOM_DIR, exist_ok=True)

    sirenler = sorted(glob.glob(os.path.join(SIREN_DIR, "*.wav")))
    nosirenler = sorted(glob.glob(os.path.join(NOSIREN_DIR, "*.wav")))
    if not sirenler or not nosirenler:
        print("HATA: data/siren veya data/no_siren bos.")
        sys.exit(1)

    siren_yolu = dosya_sec(sirenler, TERCIH_SIREN)
    no_yolu = dosya_sec(nosirenler, TERCIH_NOSIREN)

    sonuc_listesi = []
    for siren_gain, bg_gain, ad in SENARYOLAR:
        cikti_yolu = os.path.join(CUSTOM_DIR, f"{ad}.wav")
        ustuste_bindir(siren_yolu, no_yolu, siren_gain, bg_gain, cikti_yolu)

        plots_klasoru = os.path.join(PLOTS_DIR, ad)
        print(f"\n--- {ad.upper()} pipeline ---")
        sonuc = pipeline_calistir(cikti_yolu, ad, plots_klasoru)
        sonuc_listesi.append((ad, siren_gain, bg_gain, sonuc))

    # Ozet tablo
    print("\n\n" + "=" * 78)
    print("                    OVERLAY OZETI")
    print("=" * 78)
    print(f"{'Senaryo':<16} {'S:N':<10} {'is_siren':<10} {'freq_std':<10} {'autocorr':<10} {'band_E':<8} {'Conf':<6}")
    print("-" * 78)
    for ad, sg, bg, s in sonuc_listesi:
        snr = f"{sg}:{bg}"
        print(
            f"{ad:<16} {snr:<10} "
            f"{('VAR' if s['is_siren'] else 'YOK'):<10} "
            f"{s['freq_std']:<10.2f} {s['autocorr_peak']:<10.3f} "
            f"{s['band_energy_ratio']:<8.3f} {s['confidence']:<.3f}"
        )
    print("-" * 78)
    print("Esikler: freq_std>100, autocorr>0.3, band_E>0.4 (uc kosul AND)")


if __name__ == "__main__":
    main()
