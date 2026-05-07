"""
load_audio.py
-------------
WAV dosyalarini okumak icin yardimci modul.
librosa kutuphanesini kullaniyoruz cunku otomatik olarak resampling yapabiliyor.
"""

import librosa


def load_wav(filepath):
    """
    Verilen yoldaki WAV dosyasini okur ve sinyal dizisi ile sample rate dondurur.

    Parametreler:
        filepath: WAV dosyasinin tam yolu (string)

    Donus:
        signal: numpy array, ses sinyalinin ornek degerleri
        sample_rate: integer, saniyedeki ornek sayisi (Hz)
    """
    # sr=22050 diyerek dosyayi 22050 Hz'e yeniden orneklemis oluyoruz
    # boylece tum dosyalar ayni sample rate'e sahip olacak, isimiz kolaylasiyor
    # mono=True ile stereo dosyalari tek kanala indiriyoruz (ortalamasini alir)
    print(f"Ses dosyasi yukleniyor: {filepath}")

    signal, sample_rate = librosa.load(filepath, sr=22050, mono=True)

    # Yuklenen sinyal hakkinda kucuk bir bilgi
    sure = len(signal) / sample_rate
    print(f"  -> Sure: {sure:.2f} saniye, Sample Rate: {sample_rate} Hz")

    return signal, sample_rate
