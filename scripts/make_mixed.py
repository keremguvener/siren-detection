"""
make_mixed.py
-------------
Bir siren WAV ve bir no_siren WAV dosyasini arka arkaya birlestirip
'data/custom/karisik_test.wav' adi ile kaydeder, sonra ayni pipeline'i
(zaman/FFT/filtre/spektrogram/baskin frekans + detect_siren) sadece bu
karisik dosya icin calistirir.

Amac: algoritmamiz tum dosya boyunca AND mantigi ile karar verdigi icin,
icinde bir bolum siren bir bolum no_siren olan dosyada ne yaptigini gormek.

Kullanim:
    python scripts/make_mixed.py
"""

import glob
import os
import sys

# Bu betik proje kokunden cagrilabilsin diye src import path'ini ayarliyoruz.
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

# Sirayi belirler: True -> once siren sonra no_siren, False -> tersi
SIREN_ONCE = False

# Hangi siren ve no_siren'i secelim? Onceki main.py raporundan net olanlari
# tercih ediyoruz (yuksek confidence sirenden, net no_siren'den).
TERCIH_SIREN = "1-31482-B-42"   # conf 0.96 ile dogru tespit edilmisti
TERCIH_NOSIREN = "1-30709-C-23"  # net no_siren, dogru sinifanlmisti


def karisik_dosya_olustur(siren_yolu, nosiren_yolu, cikti_yolu, siren_once):
    """
    Iki WAV'i ard arda birlestirip yeni WAV olarak kaydeder.
    Sample rate'lerin ayni olmasini bekler (load_wav 22050 Hz'e resample eder).
    """
    print("Karisik dosya olusturuluyor:")
    print(f"  siren    : {os.path.basename(siren_yolu)}")
    print(f"  no_siren : {os.path.basename(nosiren_yolu)}")

    s_signal, sr_s = load_wav(siren_yolu)
    n_signal, sr_n = load_wav(nosiren_yolu)
    assert sr_s == sr_n, "Sample rate uyusmazligi"

    if siren_once:
        karisim = np.concatenate([s_signal, n_signal])
        siralama = "siren -> no_siren"
    else:
        karisim = np.concatenate([n_signal, s_signal])
        siralama = "no_siren -> siren"

    sf.write(cikti_yolu, karisim, sr_s)
    sure = len(karisim) / sr_s
    print(f"  siralama : {siralama}")
    print(f"  kaydedildi: {cikti_yolu}")
    print(f"  sure     : {sure:.2f} saniye, sr={sr_s} Hz")
    return sr_s


def pipeline_karisik(wav_yolu, plots_klasoru):
    """
    main.py'deki dosya_isle ile ayni adimlari karisik dosya icin yapar.
    """
    print("\n" + "=" * 60)
    print("KARISIK DOSYA ICIN PIPELINE")
    print("=" * 60)

    os.makedirs(plots_klasoru, exist_ok=True)

    signal, sr = load_wav(wav_yolu)

    plot_time_domain(
        signal, sr,
        title="karisik_test - Zaman Bolgesi",
        save_path=os.path.join(plots_klasoru, "01_zaman.png"),
    )

    freqs, mags = compute_fft(signal, sr)
    plot_fft(
        freqs, mags,
        title="karisik_test - FFT (Frekans Spektrumu)",
        save_path=os.path.join(plots_klasoru, "02_fft.png"),
    )
    baskin = find_dominant_frequencies(freqs, mags, top_n=5)
    print("Baskin frekanslar (ilk 5):")
    for f, m in baskin:
        print(f"   {f:.1f} Hz  (mag={m:.5f})")

    filtered = bandpass_filter(signal, sr, lowcut=500, highcut=1800, order=5)
    plot_filtered_comparison(
        signal, filtered, sr,
        title="karisik_test - Filtreleme Karsilastirmasi",
        save_path=os.path.join(plots_klasoru, "03_filtre.png"),
    )

    plot_spectrogram(
        signal, sr,
        title="karisik_test - Spektrogram",
        save_path=os.path.join(plots_klasoru, "04_spektrogram.png"),
    )

    stft_matrix, stft_freqs, stft_times = compute_stft(filtered, sr)
    dominant_freqs = track_dominant_frequency(stft_matrix, stft_freqs)
    plot_dominant_frequency_track(
        dominant_freqs, stft_times,
        title="karisik_test - Baskin Frekans Takibi",
        save_path=os.path.join(plots_klasoru, "05_baskin_frekans.png"),
    )

    sonuc = detect_siren(wav_yolu)
    return sonuc


def main():
    os.makedirs(CUSTOM_DIR, exist_ok=True)

    siren_dosyalari = sorted(glob.glob(os.path.join(SIREN_DIR, "*.wav")))
    nosiren_dosyalari = sorted(glob.glob(os.path.join(NOSIREN_DIR, "*.wav")))
    if not siren_dosyalari or not nosiren_dosyalari:
        print("HATA: data/siren veya data/no_siren bos.")
        sys.exit(1)

    # Tercih ettigimiz dosyalar varsa onu kullan, yoksa ilkini al
    siren_yolu = next(
        (p for p in siren_dosyalari if TERCIH_SIREN in p),
        siren_dosyalari[0],
    )
    nosiren_yolu = next(
        (p for p in nosiren_dosyalari if TERCIH_NOSIREN in p),
        nosiren_dosyalari[0],
    )

    cikti_yolu = os.path.join(CUSTOM_DIR, "karisik_test.wav")
    karisik_dosya_olustur(siren_yolu, nosiren_yolu, cikti_yolu, SIREN_ONCE)

    plots_klasoru = os.path.join(PLOTS_DIR, "karisik_test")
    sonuc = pipeline_karisik(cikti_yolu, plots_klasoru)

    print("\n" + "=" * 60)
    print("KARISIK DOSYA SONUCU")
    print("=" * 60)
    print(f"  is_siren          = {sonuc['is_siren']}")
    print(f"  freq_std          = {sonuc['freq_std']:.2f}  (esik > 100)")
    print(f"  autocorr_peak     = {sonuc['autocorr_peak']:.3f}  (esik > 0.3)")
    print(f"  band_energy_ratio = {sonuc['band_energy_ratio']:.3f}  (esik > 0.4)")
    print(f"  confidence        = {sonuc['confidence']:.3f}")
    print(f"\nPlots: {plots_klasoru}")


if __name__ == "__main__":
    main()
