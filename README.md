# Günlük Hayatta Siren Sesi Tespiti

Bu proje, dijital sinyal işleme (DSP) dersi kapsamında hazırlanmış öğrenci seviyesinde bir çalışmadır. Amacı, kayıt edilmiş ses dosyalarında **ambulans / polis / itfaiye sireni** gibi siren seslerini otomatik olarak tespit edebilmektir.

## Proje Açıklaması

Sirenler, düşük ve yüksek iki tonu sürekli olarak değiştiren periyodik seslerdir. Bu özelliklerden yola çıkarak ham ses sinyalini DSP teknikleriyle analiz ediyor ve "siren var" / "siren yok" şeklinde bir karar veriyoruz.

Tespitin temeli üç metriğe dayanıyor:

1. **Baskın frekansın standart sapması** → Sirende dominant frekans zamanla yukarı–aşağı oynar, std değeri yüksek olur.
2. **Otokorelasyon tepe değeri** → Siren periyodik bir desene sahiptir; otokorelasyonda belirgin bir tepe oluşur.
3. **Bant enerji oranı** → Sirenlerin enerjisi 500–1800 Hz bandında yoğunlaşır.

Üç koşul da sağlandığında dosya **"siren"** olarak etiketleniyor.

## Kullanılan DSP Yöntemleri

- **WAV dosya okuma ve resampling** (librosa)
- **Zaman bölgesi analizi** (genlik–zaman grafiği, RMS enerji)
- **Hızlı Fourier Dönüşümü (FFT)** ile frekans spektrumu
- **Butterworth Bandpass filtre** (scipy.signal) — sıfır-faz filtreleme (filtfilt)
- **Kısa Süreli Fourier Dönüşümü (STFT)** ve spektrogram
- **Baskın frekans takibi** (zaman–frekans haritasında en yüksek enerjili bin)
- **Otokorelasyon** ile periyodiklik analizi

## Klasör Yapısı

```
siren-detection/
├── data/
│   ├── siren/       # siren içeren WAV dosyaları
│   ├── no_siren/    # siren içermeyen WAV dosyaları
│   └── custom/      # etiketsiz, kendi kayıtlarınız
├── src/
│   ├── load_audio.py
│   ├── time_analysis.py
│   ├── fft_analysis.py
│   ├── filtering.py
│   ├── stft_spectrogram.py
│   └── detector.py
├── plots/           # üretilen grafikler buraya kaydedilir
├── main.py
├── requirements.txt
└── README.md
```

## Kurulum

Python 3.8 veya üzeri gerekli. Kurulum komutu:

```bash
pip install -r requirements.txt
```

Kullanılan kütüphaneler:
- numpy
- scipy
- matplotlib
- librosa
- soundfile

## Veri Seti

Eğitim/test için **UrbanSound8K** veri setinden faydalanabilirsiniz. Veri seti içinde "siren" sınıfına ait pek çok kayıt bulunuyor.

- UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html

İndirdikten sonra:
- `siren` sınıfındaki dosyaları → `data/siren/` altına
- Diğer sınıflardan (street_music, dog_bark, engine_idling, vb.) seçtiklerinizi → `data/no_siren/` altına
- Kendi telefon kayıtlarınız varsa → `data/custom/` altına

koyabilirsiniz. Tüm dosyaların `.wav` uzantılı olması gerekiyor.

## Çalıştırma

```bash
python main.py
```

Program çalışınca:
- `data/` altındaki tüm WAV dosyalarını sırayla işler.
- Her dosya için 5 farklı grafik üretip `plots/<dosya_adi>/` altına kaydeder:
  1. `01_zaman.png` — Zaman bölgesi
  2. `02_fft.png` — FFT spektrumu
  3. `03_filtre.png` — Filtre öncesi/sonrası karşılaştırma
  4. `04_spektrogram.png` — Spektrogram
  5. `05_baskin_frekans.png` — Baskın frekansın zaman içindeki değişimi
- Konsola özet tablo basar:
  ```
  Dosya Adı | Tahmin | Gerçek | Doğru mu? | Confidence
  ```
- En sonda toplam **doğruluk oranı** yazdırılır.

## Eşik Değerleri

`src/detector.py` içindeki eşikler:

| Metrik | Eşik |
|---|---|
| `freq_std` | > 100 Hz |
| `autocorr_peak` | > 0.3 |
| `band_energy_ratio` | > 0.4 |

Bu değerler veri setinize göre ince ayar yapmanıza açıktır.

## Notlar

- Bu bir öğrenci projesidir; profesyonel bir siren detektörü değildir.
- Çok kısa (< 1 saniye) veya çok gürültülü kayıtlarda yanlış sınıflandırma olabilir.
- Eşik değerleri ile oynayarak hassasiyet/özgüllük dengesini değiştirebilirsiniz.
