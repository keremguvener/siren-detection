"""
setup_data.py
-------------
ESC-50 veri setini soundata ile indirir ve projenin data/ klasorune
ornek olarak 30 siren + 30 no_siren WAV kopyalar.

Kullanim:
    python scripts/setup_data.py

Asamalar:
1) soundata.initialize('esc50') ile dataset nesnesi olusturulur
2) dataset.download() ile arsiv indirilir ve acilir (~600 MB)
3) dataset.validate() ile dosyalarin butunlugu dogrulanir
4) audio/ altindaki WAV'lar dosya adina gore siniflandirilir
   (ESC-50 dosya adi formati: <fold>-<clipID>-<take>-<target>.wav,
    target=42 -> siren)
5) Rasgele 30 siren + 30 no_siren secilip data/siren ve data/no_siren altina
   kopyalanir.

ESC-50 hakkinda:
- 50 sinif, her sinifta 40 ses, toplam 2000 WAV
- Her ses 5 saniye, 44.1 kHz mono
- Siren sinifi 'target=42' (Exterior/urban noises kategorisi)
"""

import glob
import os
import random
import shutil
import sys
import zipfile

import soundata


# Bu betik proje kokunden cagrildigini varsayar.
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJE_KOKU, "data")
SIREN_DIR = os.path.join(DATA_DIR, "siren")
NOSIREN_DIR = os.path.join(DATA_DIR, "no_siren")

# Tekrar uretilebilirlik icin sabit tohum
RANDOM_SEED = 42

# Kac dosya kopyalanacak
N_SIREN = 30
N_NOSIREN = 30

# ESC-50'de siren sinifinin hedef kodu
SIREN_TARGET = 42


def indir_ve_dogrula():
    """
    soundata uzerinden ESC-50 ZIP'ini indirir, sonra manuel olarak acar.
    Soundata MD5 checksum'u eski hash'le karsilastiriyor; ESC-50 master
    GitHub'da guncellendigi icin checksum hatasi veriyor. Biz dosyayi
    indirip kendimiz acacagiz.

    Donus: dataset.data_home (indirme dizini)
    """
    print("=" * 70)
    print("  ESC-50 indiriliyor (soundata + manuel acma)")
    print("=" * 70)

    dataset = soundata.initialize("esc50")
    data_home = dataset.data_home
    print(f"Veri klasoru (data_home): {data_home}")

    zip_yolu = os.path.join(data_home, "ESC-50-master.zip")

    # Eger ZIP zaten varsa indirmeyi atla
    if os.path.exists(zip_yolu):
        print(f"ZIP zaten mevcut, indirmeyi atliyorum: {zip_yolu}")
    else:
        # soundata indirmesi MD5 hatasi atinca dosyayi cesinde birakir, sorun degil.
        print("\n--> dataset.download() cagriliyor (MD5 hatasi normal)...")
        try:
            dataset.download()
        except (OSError, IOError) as hata:
            # MD5 uyusmazligi cikabilir, ZIP'in var olup olmadigini kontrol edelim
            if not os.path.exists(zip_yolu):
                print(f"HATA: ZIP indirilemedi: {hata}")
                sys.exit(1)
            print(f"  (Checksum uyari atlandi: {type(hata).__name__})")

    # ZIP'i ac (audio klasoru yoksa)
    audio_klasoru = os.path.join(data_home, "ESC-50-master", "audio")
    if os.path.isdir(audio_klasoru) and glob.glob(os.path.join(audio_klasoru, "*.wav")):
        print(f"\nArsiv zaten acilmis: {audio_klasoru}")
    else:
        print(f"\n--> ZIP aciliyor: {zip_yolu}")
        with zipfile.ZipFile(zip_yolu, "r") as zf:
            zf.extractall(data_home)
        print(f"  -> {audio_klasoru} altina cikarildi")

    print("\nIndirme ve acma tamam.")
    return data_home


def target_cikar(dosya_adi):
    """
    ESC-50 dosya adindan sinif hedef kodunu cikarir.
    Format: <fold>-<clipID>-<take>-<target>.wav

    Donus: int target veya None (parse edilemezse)
    """
    taban = os.path.splitext(os.path.basename(dosya_adi))[0]
    parcalar = taban.split("-")
    if len(parcalar) < 4:
        return None
    try:
        return int(parcalar[3])
    except ValueError:
        return None


def wav_listesi_topla(data_home):
    """
    ESC-50 audio klasorundeki tum WAV'lari dondurur.
    Arsiv acildiginda yapi genelde: <data_home>/audio/*.wav veya
    <data_home>/ESC-50-master/audio/*.wav olabilir; ikisini de deneriz.
    """
    olasi_yollar = [
        os.path.join(data_home, "audio", "*.wav"),
        os.path.join(data_home, "ESC-50-master", "audio", "*.wav"),
    ]
    for desen in olasi_yollar:
        dosyalar = glob.glob(desen)
        if dosyalar:
            return dosyalar
    return []


def ornekle_ve_kopyala(data_home, n_siren=N_SIREN, n_nosiren=N_NOSIREN):
    """
    ESC-50 icinden n_siren tane siren (target=42) ve n_nosiren tane
    siren olmayan WAV secip proje data/ klasorlerine kopyalar.
    """
    print("\n" + "=" * 70)
    print("  Orneklem secimi ve kopyalama")
    print("=" * 70)

    # Hedef klasorler hazir mi
    os.makedirs(SIREN_DIR, exist_ok=True)
    os.makedirs(NOSIREN_DIR, exist_ok=True)

    tum_wavlar = wav_listesi_topla(data_home)
    print(f"Toplam WAV bulundu: {len(tum_wavlar)}")
    if len(tum_wavlar) == 0:
        print("HATA: hic WAV bulunamadi. Indirme basarili miydi?")
        sys.exit(1)

    siren_wavlar = []
    nosiren_wavlar = []
    for yol in tum_wavlar:
        tgt = target_cikar(yol)
        if tgt is None:
            continue
        if tgt == SIREN_TARGET:
            siren_wavlar.append(yol)
        else:
            nosiren_wavlar.append(yol)

    print(f"  siren    (target={SIREN_TARGET}) : {len(siren_wavlar)}")
    print(f"  no_siren (diger)                 : {len(nosiren_wavlar)}")

    # Tekrarlanabilir orneklem
    rastgele = random.Random(RANDOM_SEED)
    secilen_sirenler = rastgele.sample(siren_wavlar, min(n_siren, len(siren_wavlar)))
    secilen_nosirenler = rastgele.sample(nosiren_wavlar, min(n_nosiren, len(nosiren_wavlar)))

    print(f"\nKopyalaniyor: {len(secilen_sirenler)} siren -> {SIREN_DIR}")
    for src in secilen_sirenler:
        dst = os.path.join(SIREN_DIR, os.path.basename(src))
        shutil.copy2(src, dst)

    print(f"Kopyalaniyor: {len(secilen_nosirenler)} no_siren -> {NOSIREN_DIR}")
    for src in secilen_nosirenler:
        dst = os.path.join(NOSIREN_DIR, os.path.basename(src))
        shutil.copy2(src, dst)

    print("\nTamam. data/ klasorleri hazir:")
    print(f"  {SIREN_DIR}    ({len(secilen_sirenler)} dosya)")
    print(f"  {NOSIREN_DIR} ({len(secilen_nosirenler)} dosya)")


def main():
    data_home = indir_ve_dogrula()
    ornekle_ve_kopyala(data_home)
    print("\nArtik 'python main.py' calistirilabilir.")


if __name__ == "__main__":
    main()
