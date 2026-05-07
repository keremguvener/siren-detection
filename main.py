"""
main.py
-------
Tum projeyi calistiran ana dosya.

Yaptigi isler:
- data/siren ve data/no_siren klasorlerindeki tum WAV dosyalarini bulur
- Her dosya icin:
    * Zaman bolgesi grafigi
    * FFT grafigi
    * Filtreli/filtresiz karsilastirma grafigi
    * Spektrogram
    * Baskin frekans takip grafigi
    * Tespit sonucu (SIREN VAR / YOK)
- Tum grafikleri plots/ altina kaydeder
- Konsola ozet tablo basar
- Sonunda toplam dogruluk oranini yazar
"""

import os
import glob

# Projedeki modulleri import ediyoruz
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


# Klasor yollari
DATA_DIR = "data"
PLOTS_DIR = "plots"


def dosya_isle(wav_yolu, gercek_etiket):
    """
    Tek bir WAV dosyasi icin tum analiz adimlarini calistirir.

    Parametreler:
        wav_yolu: WAV dosyasinin yolu
        gercek_etiket: 'siren' / 'no_siren' / 'unknown' (gercek etiket)

    Donus:
        satir: dict, ozet tablo icin tek satir bilgi
    """
    # Dosya adindan grafik isimleri uretelim
    dosya_adi = os.path.splitext(os.path.basename(wav_yolu))[0]

    # Plots klasorunde her dosya icin alt klasor olusturalim (kalabaligi onlemek icin)
    cikti_klasoru = os.path.join(PLOTS_DIR, dosya_adi)
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)

    print("\n" + "#" * 70)
    print(f"# Isleniyor: {wav_yolu}")
    print(f"# Gercek etiket: {gercek_etiket}")
    print("#" * 70)

    try:
        # 1) Sinyali yukle
        signal, sr = load_wav(wav_yolu)

        # 2) Zaman bolgesi grafigi
        plot_time_domain(
            signal, sr,
            title=f"{dosya_adi} - Zaman Bolgesi",
            save_path=os.path.join(cikti_klasoru, "01_zaman.png")
        )

        # 3) FFT analizi
        freqs, mags = compute_fft(signal, sr)
        plot_fft(
            freqs, mags,
            title=f"{dosya_adi} - FFT (Frekans Spektrumu)",
            save_path=os.path.join(cikti_klasoru, "02_fft.png")
        )
        # Baskin frekanslari yazdir
        baskin = find_dominant_frequencies(freqs, mags, top_n=5)
        print("Baskin frekanslar (ilk 5):")
        for f, m in baskin:
            print(f"   {f:.1f} Hz  (mag={m:.5f})")

        # 4) Filtreleme ve karsilastirma
        filtered = bandpass_filter(signal, sr, lowcut=500, highcut=1800, order=5)
        plot_filtered_comparison(
            signal, filtered, sr,
            title=f"{dosya_adi} - Filtreleme Karsilastirmasi",
            save_path=os.path.join(cikti_klasoru, "03_filtre.png")
        )

        # 5) Spektrogram
        plot_spectrogram(
            signal, sr,
            title=f"{dosya_adi} - Spektrogram",
            save_path=os.path.join(cikti_klasoru, "04_spektrogram.png")
        )

        # 6) Baskin frekans takibi
        stft_matrix, stft_freqs, stft_times = compute_stft(filtered, sr)
        dominant_freqs = track_dominant_frequency(stft_matrix, stft_freqs)
        plot_dominant_frequency_track(
            dominant_freqs, stft_times,
            title=f"{dosya_adi} - Baskin Frekans Takibi",
            save_path=os.path.join(cikti_klasoru, "05_baskin_frekans.png")
        )

        # 7) Siren tespiti
        sonuc = detect_siren(wav_yolu)

        # Tahmin etiketini olustur
        tahmin_etiketi = "siren" if sonuc['is_siren'] else "no_siren"

        # Dogru tahmin mi?
        if gercek_etiket == "unknown":
            dogru_mu = "?"
        else:
            dogru_mu = "EVET" if tahmin_etiketi == gercek_etiket else "HAYIR"

        return {
            'dosya': dosya_adi,
            'tahmin': tahmin_etiketi,
            'gercek': gercek_etiket,
            'dogru': dogru_mu,
            'confidence': sonuc['confidence']
        }

    except Exception as hata:
        # Cok detayli hata yakalamayalim, basitce bildirelim
        print(f"HATA: {wav_yolu} islenirken sorun olustu -> {hata}")
        return {
            'dosya': dosya_adi,
            'tahmin': "HATA",
            'gercek': gercek_etiket,
            'dogru': "?",
            'confidence': 0.0
        }


def main():
    """
    Programin giris noktasi. data/ klasorundeki butun dosyalari isler.
    """
    print("=" * 70)
    print("  GUNLUK HAYATTA SIREN SESI TESPITI - DSP PROJESI")
    print("=" * 70)

    # Plots klasorunun varligini garantile
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # data/siren altinda olan dosyalar siren etiketli
    siren_dosyalari = glob.glob(os.path.join(DATA_DIR, "siren", "*.wav"))
    # data/no_siren altinda olan dosyalar siren olmayanlar
    nosiren_dosyalari = glob.glob(os.path.join(DATA_DIR, "no_siren", "*.wav"))
    # data/custom altinda etiket bilinmiyor
    custom_dosyalari = glob.glob(os.path.join(DATA_DIR, "custom", "*.wav"))

    print(f"\nBulunan dosya sayilari:")
    print(f"  siren:    {len(siren_dosyalari)}")
    print(f"  no_siren: {len(nosiren_dosyalari)}")
    print(f"  custom:   {len(custom_dosyalari)}")

    if (len(siren_dosyalari) + len(nosiren_dosyalari) + len(custom_dosyalari)) == 0:
        print("\nUYARI: data/ klasorunde hic WAV dosyasi yok.")
        print("Lutfen data/siren/ ve data/no_siren/ altina bazi WAV dosyalari koyun.")
        print("(README.md icinde UrbanSound8K linki var.)")
        return

    # Tum dosyalari isle
    sonuclar = []

    for dosya in siren_dosyalari:
        sonuclar.append(dosya_isle(dosya, "siren"))

    for dosya in nosiren_dosyalari:
        sonuclar.append(dosya_isle(dosya, "no_siren"))

    for dosya in custom_dosyalari:
        sonuclar.append(dosya_isle(dosya, "unknown"))

    # ----- OZET TABLO -----
    print("\n\n" + "=" * 70)
    print("                          OZET TABLO")
    print("=" * 70)
    print(f"{'Dosya Adi':<30} {'Tahmin':<12} {'Gercek':<12} {'Dogru?':<8} {'Conf':<6}")
    print("-" * 70)

    dogru_sayisi = 0
    bilinen_toplam = 0

    for s in sonuclar:
        print(
            f"{s['dosya'][:28]:<30} {s['tahmin']:<12} {s['gercek']:<12} {s['dogru']:<8} {s['confidence']:.2f}"
        )
        if s['dogru'] == "EVET":
            dogru_sayisi += 1
        if s['gercek'] != "unknown" and s['tahmin'] != "HATA":
            bilinen_toplam += 1

    # Dogruluk oranini yazdir
    print("-" * 70)
    if bilinen_toplam > 0:
        dogruluk = dogru_sayisi / bilinen_toplam
        print(f"\nToplam etiketli dosya: {bilinen_toplam}")
        print(f"Dogru tahmin sayisi:   {dogru_sayisi}")
        print(f"DOGRULUK ORANI: {dogruluk * 100:.2f}%")
    else:
        print("\nEtiketli (siren/no_siren) dosya bulunamadi, dogruluk hesaplanamadi.")

    print("\nTum grafikler 'plots/' klasorune kaydedildi.")
    print("Bitti.")


if __name__ == "__main__":
    main()
