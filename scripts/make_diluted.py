"""
make_diluted.py
---------------
Seyreltilmis test: 3 dosya ard arda eklenir.
Siralama: no_siren_A -> siren -> no_siren_B (toplam 15 saniye, ~%33 siren)

Amac: siren parcasi dosyanin sadece 1/3'unu kapladiginda algoritmamiz hala
'siren var' diyebiliyor mu? Confidence ne kadar dusuyor? AND mantigi
hangi metrikten patliyor (eger patliyor)?

Kullanim:
    python scripts/make_diluted.py
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

# Net bir siren (onceki testten conf 0.96)
TERCIH_SIREN = "1-31482-B-42"
# Iki farkli no_siren secelim, cesitlilik icin
TERCIH_NOSIREN_A = "1-30709-C-23"   # breathing
TERCIH_NOSIREN_B = "4-159609-A-14"  # chirping_birds (TP'lerimizden)

# Sirenin yerlesimi: 'middle' (sandwich), 'start', 'end'
SIREN_POZISYON = "middle"

CIKTI_AD = "karisik_seyrek.wav"


def dosya_sec(dosyalar, anahtar, varsayilan_idx=0):
    """Liste icinden istenen ismi iceren dosyayi sec, yoksa varsayilan."""
    for p in dosyalar:
        if anahtar in p:
            return p
    return dosyalar[varsayilan_idx]


def karisik_olustur(siren_yolu, no_a_yolu, no_b_yolu, cikti_yolu, pozisyon):
    """
    Uc dosyayi pozisyona gore birlestirip kaydeder.
    """
    print("Seyreltilmis karisik dosya olusturuluyor:")
    print(f"  siren     : {os.path.basename(siren_yolu)}")
    print(f"  no_siren A: {os.path.basename(no_a_yolu)}")
    print(f"  no_siren B: {os.path.basename(no_b_yolu)}")

    s, sr_s = load_wav(siren_yolu)
    a, sr_a = load_wav(no_a_yolu)
    b, sr_b = load_wav(no_b_yolu)
    assert sr_s == sr_a == sr_b, "Sample rate uyusmazligi"

    if pozisyon == "middle":
        karisim = np.concatenate([a, s, b])
        siralama = "no_siren_A -> SIREN -> no_siren_B"
    elif pozisyon == "start":
        karisim = np.concatenate([s, a, b])
        siralama = "SIREN -> no_siren_A -> no_siren_B"
    elif pozisyon == "end":
        karisim = np.concatenate([a, b, s])
        siralama = "no_siren_A -> no_siren_B -> SIREN"
    else:
        raise ValueError(f"Bilinmeyen pozisyon: {pozisyon}")

    sf.write(cikti_yolu, karisim, sr_s)
    sure = len(karisim) / sr_s
    siren_orani = len(s) / len(karisim) * 100
    print(f"  pozisyon  : {pozisyon} ({siralama})")
    print(f"  toplam    : {sure:.2f} saniye, siren orani: %{siren_orani:.1f}")
    print(f"  kaydedildi: {cikti_yolu}")
    return sr_s


def pipeline_calistir(wav_yolu, baslik, plots_klasoru):
    """main.py'deki dosya_isle ile ayni adimlar."""
    print("\n" + "=" * 60)
    print(f"PIPELINE: {baslik}")
    print("=" * 60)

    os.makedirs(plots_klasoru, exist_ok=True)
    signal, sr = load_wav(wav_yolu)

    plot_time_domain(signal, sr, f"{baslik} - Zaman Bolgesi",
                     os.path.join(plots_klasoru, "01_zaman.png"))

    freqs, mags = compute_fft(signal, sr)
    plot_fft(freqs, mags, f"{baslik} - FFT",
             os.path.join(plots_klasoru, "02_fft.png"))
    baskin = find_dominant_frequencies(freqs, mags, top_n=5)
    print("Baskin frekanslar (ilk 5):")
    for f, m in baskin:
        print(f"   {f:.1f} Hz  (mag={m:.5f})")

    filtered = bandpass_filter(signal, sr, lowcut=500, highcut=1800, order=5)
    plot_filtered_comparison(signal, filtered, sr,
                             f"{baslik} - Filtreleme Karsilastirmasi",
                             os.path.join(plots_klasoru, "03_filtre.png"))

    plot_spectrogram(signal, sr, f"{baslik} - Spektrogram",
                     os.path.join(plots_klasoru, "04_spektrogram.png"))

    stft_matrix, ff, tt = compute_stft(filtered, sr)
    dom = track_dominant_frequency(stft_matrix, ff)
    plot_dominant_frequency_track(dom, tt, f"{baslik} - Baskin Frekans Takibi",
                                  os.path.join(plots_klasoru, "05_baskin_frekans.png"))

    return detect_siren(wav_yolu)


def main():
    os.makedirs(CUSTOM_DIR, exist_ok=True)

    sirenler = sorted(glob.glob(os.path.join(SIREN_DIR, "*.wav")))
    nosirenler = sorted(glob.glob(os.path.join(NOSIREN_DIR, "*.wav")))
    if not sirenler or len(nosirenler) < 2:
        print("HATA: yeterli dosya yok (en az 1 siren + 2 no_siren).")
        sys.exit(1)

    siren_yolu = dosya_sec(sirenler, TERCIH_SIREN)
    no_a = dosya_sec(nosirenler, TERCIH_NOSIREN_A, 0)
    no_b = dosya_sec(nosirenler, TERCIH_NOSIREN_B, 1)

    cikti_yolu = os.path.join(CUSTOM_DIR, CIKTI_AD)
    karisik_olustur(siren_yolu, no_a, no_b, cikti_yolu, SIREN_POZISYON)

    plots_klasoru = os.path.join(PLOTS_DIR, os.path.splitext(CIKTI_AD)[0])
    sonuc = pipeline_calistir(cikti_yolu, "karisik_seyrek", plots_klasoru)

    print("\n" + "=" * 60)
    print("SEYRELTILMIS KARISIK SONUC")
    print("=" * 60)
    durum = "SIREN VAR" if sonuc['is_siren'] else "SIREN YOK"
    print(f"  Sonuc             : {durum}")
    print(f"  freq_std          = {sonuc['freq_std']:.2f}  (esik > 100)")
    print(f"  autocorr_peak     = {sonuc['autocorr_peak']:.3f}  (esik > 0.3)")
    print(f"  band_energy_ratio = {sonuc['band_energy_ratio']:.3f}  (esik > 0.4)")
    print(f"  confidence        = {sonuc['confidence']:.3f}")
    print(f"\nPlots: {plots_klasoru}")


if __name__ == "__main__":
    main()
