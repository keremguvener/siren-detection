"""
make_scenario.py
----------------
Gercekci senaryo: Koşucu ormanda kosuyor, arkadan ambulans yaklasiyor.

Ses katmanlari (hepsi ust uste, 5 saniyelik klipler):
  - chirping_birds  : orman atmosferi (sabit, orta seviye)
  - breathing       : kosucunun nefesi (sabit, dusuk-orta seviye)
  - siren           : yaklasian ambulans (fade-in: 0.15 -> 1.0)

Karistirma:
  signal = bg_birds * 0.45 + bg_nefes * 0.40 + siren * fade_in_zarfi
  -> sonra clipping olmasin diye |max|'a bolerek normalize edilir.

Amac: Algoritmamiz gercek hayatta da ambulansi yakalayabilir mi?
Eger evet ise, bu teknoloji "kosarken kulaklikla muzik dinlerken arkadan
siren geldigini sana haber veririm" diye konumlandirilabilir.

Birden fazla yaklasim seviyesi de uretiriz:
  - 'uzak'   : siren fade-in 0.05 -> 0.40  (cok uzakta)
  - 'orta'   : siren fade-in 0.15 -> 0.80  (yaklasiyor)
  - 'yakin'  : siren fade-in 0.30 -> 1.00  (cok yakin)

Boylece "kac saniye uzakta yakalayabiliriz" sorusunu da gosteririz.

Kullanim:
    python scripts/make_scenario.py
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

# Net dosyalar (onceki testlerden)
TERCIH_SIREN = "1-31482-B-42"
TERCIH_KUS = "4-159609-A-14"      # chirping_birds (target=14)
TERCIH_NEFES = "1-30709-C-23"     # breathing (target=23)

# Arkaplan kazanc seviyeleri (sabit)
BG_KUS_GAIN = 0.45
BG_NEFES_GAIN = 0.40

# (ad, fade_baslangic, fade_son) - sirenin yaklasma profilleri
SENARYOLAR = [
    ("senaryo_uzak", 0.05, 0.40),
    ("senaryo_orta", 0.15, 0.80),
    ("senaryo_yakin", 0.30, 1.00),
]


def dosya_sec(dosyalar, anahtar):
    for p in dosyalar:
        if anahtar in p:
            return p
    return dosyalar[0]


def fade_in_uygula(sinyal, baslangic_gain, son_gain):
    """
    Lineer fade-in zarfi uygular: gain baslangic_gain'den son_gain'e
    sinyal boyunca esit artarak gider.
    """
    n = len(sinyal)
    zarf = np.linspace(baslangic_gain, son_gain, n)
    return sinyal * zarf


def senaryo_olustur(siren_yolu, kus_yolu, nefes_yolu,
                    fade_bas, fade_son, cikti_yolu):
    """
    Uc katmani toplama ile birlestirir, clipping'i onlemek icin normalize edip
    kaydeder.
    """
    print(f"\n[SENARYO] siren fade-in: {fade_bas:.2f} -> {fade_son:.2f}")
    print(f"  siren  : {os.path.basename(siren_yolu)}")
    print(f"  kus    : {os.path.basename(kus_yolu)}  gain={BG_KUS_GAIN}")
    print(f"  nefes  : {os.path.basename(nefes_yolu)} gain={BG_NEFES_GAIN}")

    s, sr_s = load_wav(siren_yolu)
    k, sr_k = load_wav(kus_yolu)
    n, sr_n = load_wav(nefes_yolu)
    assert sr_s == sr_k == sr_n, "Sample rate uyusmazligi"

    # Hepsini min uzunluga kirp (guvenli)
    L = min(len(s), len(k), len(n))
    s = s[:L]
    k = k[:L]
    n = n[:L]

    # Siren'e yaklasma zarfi uygula
    s_faded = fade_in_uygula(s, fade_bas, fade_son)

    # Toplam karisim
    karisim = BG_KUS_GAIN * k + BG_NEFES_GAIN * n + s_faded

    # Clipping onlemek icin normalize
    tepe = np.max(np.abs(karisim))
    if tepe > 0:
        karisim = karisim / tepe * 0.99

    sf.write(cikti_yolu, karisim, sr_s)
    print(f"  kaydedildi: {cikti_yolu}  ({L/sr_s:.2f} s)")
    return sr_s


def pipeline_calistir(wav_yolu, baslik, plots_klasoru):
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
    if not sirenler or len(nosirenler) < 2:
        print("HATA: yeterli dosya yok.")
        sys.exit(1)

    siren_yolu = dosya_sec(sirenler, TERCIH_SIREN)
    kus_yolu = dosya_sec(nosirenler, TERCIH_KUS)
    nefes_yolu = dosya_sec(nosirenler, TERCIH_NEFES)

    print("=" * 70)
    print("  SENARYO: Ormanda Kosucu + Yaklasian Ambulans")
    print("=" * 70)
    print("  Arkaplan : kus sesi + nefes (sabit)")
    print("  On plan  : siren (fade-in ile yaklasiyor)")
    print("  3 mesafe denenir: uzak / orta / yakin")

    sonuc_listesi = []
    for ad, fb, fs in SENARYOLAR:
        cikti_yolu = os.path.join(CUSTOM_DIR, f"{ad}.wav")
        senaryo_olustur(siren_yolu, kus_yolu, nefes_yolu, fb, fs, cikti_yolu)

        plots_klasoru = os.path.join(PLOTS_DIR, ad)
        print(f"\n--- {ad.upper()} pipeline ---")
        sonuc = pipeline_calistir(cikti_yolu, ad, plots_klasoru)
        sonuc_listesi.append((ad, fb, fs, sonuc))

    # Ozet
    print("\n\n" + "=" * 86)
    print("                       SENARYO OZETI")
    print("=" * 86)
    print(f"{'Senaryo':<18} {'Siren fade':<14} {'Tespit':<10} {'freq_std':<10} {'autocorr':<10} {'band_E':<8} {'Conf':<6}")
    print("-" * 86)
    for ad, fb, fs, s in sonuc_listesi:
        fade = f"{fb:.2f}->{fs:.2f}"
        durum = "VAR" if s['is_siren'] else "YOK"
        print(
            f"{ad:<18} {fade:<14} {durum:<10} "
            f"{s['freq_std']:<10.2f} {s['autocorr_peak']:<10.3f} "
            f"{s['band_energy_ratio']:<8.3f} {s['confidence']:<.3f}"
        )
    print("-" * 86)
    print("Esikler: freq_std>100, autocorr>0.3, band_E>0.4")


if __name__ == "__main__":
    main()
