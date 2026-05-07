# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje Özeti

Bu, **DSP dersi öğrenci ödevi** olarak yazılmış bir Türkçe siren tespit projesidir (`siren-detection/` altında). Profesyonel/üretim kodu değildir; bilinçli olarak öğrenci tarzı tutulmuştur. Kod stiline dair karar verirken bu bağlamı koru.

## Çalıştırma

```powershell
cd siren-detection
pip install -r requirements.txt
python main.py
```

`main.py`'nin tek giriş noktası vardır; testler veya lint yapılandırması yoktur. Tek dosya işlemek için yeni bir script yazmak yerine `data/custom/` içine kopyalayıp `main.py` çalıştırılır.

## Mimari (büyük resim)

Pipeline `src/detector.py` → `detect_siren(filepath)` etrafında döner ve **üç metriğin AND'lenmesiyle** karar verir:

1. **Bandpass filtre** (500-1800 Hz, Butterworth + filtfilt) → sinyali siren bandına odaklar (`src/filtering.py`)
2. **STFT** ile zaman-frekans matrisi (`src/stft_spectrogram.py`)
3. Her zaman dilimindeki **dominant frekans serisi** çıkarılır (`track_dominant_frequency`)
4. Bu seriden 3 metrik:
   - `freq_std > 100` → siren frekansı salınır
   - `autocorr_peak > 0.3` → periyodik yapı (otokorelasyon `lag>=5`'ten itibaren bakılır)
   - `band_energy_ratio > 0.4` → enerji 500-1800 Hz'de yoğunlaşmış (orijinal sinyal üzerinden, filtreli değil)

Üçü de geçerse `is_siren=True`. Threshold değerlerini değiştirme isteği gelirse kaynak: `src/detector.py`.

`main.py` her dosya için sırasıyla `time_analysis → fft_analysis → filtering → stft_spectrogram → detector` modüllerini çağırıp `plots/<dosya_adi>/` altına 5 PNG yazar, sonunda etiketli dosyalar üzerinden doğruluk oranı hesaplar.

## Veri Düzeni

`main.py` dosyaları **klasör ismine göre etiketler**:

- `data/siren/*.wav` → ground truth = siren
- `data/no_siren/*.wav` → ground truth = no_siren
- `data/custom/*.wav` → etiket bilinmiyor (`unknown`), doğruluğa katılmaz

Önerilen veri kaynağı: **UrbanSound8K**. Dosya adında ortadaki rakam sınıf ID'sidir; `8` = siren. Kullanıcının yönlendirildiği yerleştirme: ~10-30 siren + ~10-30 no_siren + birkaç custom.

## Kod Stili (kasıtlı kararlar — değiştirme)

Bu kurallar öğrenci kodu havasını korumak için bilerek seçildi:

- **Türkçe docstring ve yorum** (ASCII, Türkçe karakter yok — Windows konsol uyumu için)
- **Type hint YOK**
- **Emoji YOK** (dosyalarda da, çıktıda da)
- Değişken isimleri **Türkçe-İngilizce karışık** olabilir (`baskin_frekanslar`, `cikti_klasoru`, `signal`, `freqs`)
- `print()` ile bol bilgilendirme ("FFT hesaplaniyor...", "Filtre uygulandi...") — kullanıcı çıktıyı görüyor, bu istenen davranış
- Hata yakalama **basit** (`main.py`'de tek bir geniş `try/except`); detaylı hata sınıfı hiyerarşisi ekleme
- Karmaşık satırların üstünde *ne yaptığını* açıklayan yorum var — bu üslubu sürdür

Yeni modül eklerken aynı tonu koru: kısa Türkçe docstring + her fonksiyonun başında "ne yapıyor + hangi parametreler + ne dönüyor" formatı.

## Sık Karşılaşılan Değişiklik İstekleri

- **Threshold ayarı:** `src/detector.py` içinde `threshold_std`, `threshold_autocorr`, `threshold_band` değişkenleri.
- **Bandpass aralığı:** `src/filtering.py` `bandpass_filter` default'ları ve `detector.py`'deki `hesapla_bant_enerji_orani` çağrısı — **iki yeri birden** güncellemek gerekir, yoksa filtre ile enerji oranı ölçümü uyuşmaz.
- **Frekans grafiği aralığı:** `src/fft_analysis.py` ve `stft_spectrogram.py` içinde `4000` Hz sabit kodlu — siren bandı buralarda değişmediği için sabit tutuldu.
- **Sample rate:** `src/load_audio.py` `sr=22050` sabit; tüm pipeline bunu varsayar.

## Platform Notları

- Windows + PowerShell hedef ortam.
- librosa kurulumu Windows'ta zorlanırsa `pip install --upgrade pip setuptools wheel` sonra tekrar dene.
- `plots/` her çalıştırmada üzerine yazar (alt klasör isimleri dosya adından gelir).
