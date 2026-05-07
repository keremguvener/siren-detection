"""
detector.py
-----------
Asil siren tespit algoritmasi.

Mantik:
- Once sinyali bandpass filtreden geciriyoruz (siren bandi: 500-1800 Hz)
- STFT ile zaman-frekans analizi yapiyoruz
- Her zaman diliminde baskin frekans neyse onu kaydediyoruz
- Bu baskin frekans serisinin uc ozelligine bakiyoruz:
    1) Standart sapma -> siren frekansi salinim yapar, std yuksek olur
    2) Otokorelasyon tepe degeri -> siren periyodiktir
    3) Bant enerji orani -> enerjinin ne kadari siren bandinda?
- Uc kosul da saglanirsa SIREN VAR diyoruz.
"""

import numpy as np

from src.load_audio import load_wav
from src.filtering import bandpass_filter
from src.stft_spectrogram import compute_stft, track_dominant_frequency


def hesapla_otokorelasyon_tepe(seri):
    """
    Bir zaman serisinin otokorelasyonunu hesaplar ve en yuksek tepe degerini dondurur.
    Periyodik sinyallerde otokorelasyonda belirgin tepeler olusur.

    Parametreler:
        seri: 1D numpy array (baskin frekans zaman serisi)

    Donus:
        max_tepe: float, lag>0 icin en yuksek normalize otokorelasyon
    """
    # Ortalamayi cikariyoruz ki DC bileseni etkisiz olsun
    seri_merkez = seri - np.mean(seri)

    # numpy correlate ile tam otokorelasyon
    autocorr = np.correlate(seri_merkez, seri_merkez, mode='full')

    # Sadece pozitif lag kismini al (ortadan saga)
    autocorr = autocorr[len(autocorr) // 2:]

    # 0. lag her zaman en buyuktur (sinyal kendisiyle tam ortuser), normalize et
    if autocorr[0] == 0:
        return 0.0
    autocorr = autocorr / autocorr[0]

    # Cok kisa lag'lerde hala 1'e yakin degerler olur, biz biraz ileriden bakalim
    # En az 5 cerceve sonrasinda tepe ariyoruz (bu kisa periyodlari atlar)
    if len(autocorr) <= 5:
        return 0.0

    max_tepe = np.max(autocorr[5:])
    return float(max_tepe)


def hesapla_bant_enerji_orani(signal, sr, lowcut=500, highcut=1800):
    """
    Sinyal enerjisinin ne kadarinin [lowcut, highcut] bandinda oldugunu bulur.
    Sirende bu oran yuksek olur cunku tum enerji o bandda yogunlasir.

    Parametreler:
        signal: ses sinyali
        sr: sample rate
        lowcut, highcut: bant sinirlari (Hz)

    Donus:
        oran: float (0-1 arasi)
    """
    # FFT al
    N = len(signal)
    fft_sonucu = np.fft.fft(signal)
    mags = np.abs(fft_sonucu[:N // 2])
    freqs = np.fft.fftfreq(N, d=1.0 / sr)[:N // 2]

    # Toplam enerji (magnitude karelerinin toplami)
    toplam_enerji = np.sum(mags ** 2)
    if toplam_enerji == 0:
        return 0.0

    # Siren bandindaki enerji
    bant_mask = (freqs >= lowcut) & (freqs <= highcut)
    bant_enerji = np.sum(mags[bant_mask] ** 2)

    oran = bant_enerji / toplam_enerji
    return float(oran)


def detect_siren(filepath):
    """
    Verilen WAV dosyasinda siren olup olmadigini tespit eder.

    Parametreler:
        filepath: WAV dosyasinin yolu

    Donus:
        sonuc: dict, {
            'is_siren': True/False,
            'freq_std': baskin frekans serisinin std'i,
            'autocorr_peak': otokorelasyon tepe degeri,
            'band_energy_ratio': siren bandindaki enerji orani,
            'confidence': 0-1 arasi guven skoru
        }
    """
    print("\n" + "=" * 60)
    print(f"SIREN TESPITI BASLIYOR: {filepath}")
    print("=" * 60)

    # 1) Dosyayi yukle
    signal, sr = load_wav(filepath)

    # 2) Bandpass filtre uygula (siren bandina odaklan)
    filtered = bandpass_filter(signal, sr, lowcut=500, highcut=1800, order=5)

    # 3) STFT al
    stft_matrix, freqs, times = compute_stft(filtered, sr)

    # 4) Her zaman dilimi icin baskin frekansi bul
    dominant_freqs = track_dominant_frequency(stft_matrix, freqs)

    # 5) Metrikleri hesapla
    # Std: baskin frekansin ne kadar oynadigi
    freq_std = float(np.std(dominant_freqs))

    # Otokorelasyon: periyodiklik var mi?
    autocorr_peak = hesapla_otokorelasyon_tepe(dominant_freqs)

    # Bant enerji orani (orijinal sinyal uzerinden)
    band_energy_ratio = hesapla_bant_enerji_orani(signal, sr, 500, 1800)

    # 6) Threshold'lara gore karar ver
    threshold_std = 100.0
    threshold_autocorr = 0.3
    threshold_band = 0.4

    kosul1 = freq_std > threshold_std
    kosul2 = autocorr_peak > threshold_autocorr
    kosul3 = band_energy_ratio > threshold_band

    is_siren = bool(kosul1 and kosul2 and kosul3)

    # Confidence: kac threshold'u gectigini ve ne kadar gectigini ortalayarak
    # basit bir guven skoru olusturalim (0-1 arasi)
    skor1 = min(freq_std / threshold_std, 2.0) / 2.0
    skor2 = min(autocorr_peak / threshold_autocorr, 2.0) / 2.0
    skor3 = min(band_energy_ratio / threshold_band, 2.0) / 2.0
    confidence = float((skor1 + skor2 + skor3) / 3.0)

    # Sonuclari yazdir
    print("\n--- METRIKLER ---")
    print(f"  freq_std           = {freq_std:.2f}   (esik > {threshold_std})  -> {'OK' if kosul1 else 'X'}")
    print(f"  autocorr_peak      = {autocorr_peak:.3f}  (esik > {threshold_autocorr})   -> {'OK' if kosul2 else 'X'}")
    print(f"  band_energy_ratio  = {band_energy_ratio:.3f}  (esik > {threshold_band})   -> {'OK' if kosul3 else 'X'}")
    print(f"  confidence         = {confidence:.3f}")
    print(f"\n  SONUC: {'SIREN TESPIT EDILDI' if is_siren else 'SIREN YOK'}")

    sonuc = {
        'is_siren': is_siren,
        'freq_std': freq_std,
        'autocorr_peak': autocorr_peak,
        'band_energy_ratio': band_energy_ratio,
        'confidence': confidence
    }

    return sonuc
