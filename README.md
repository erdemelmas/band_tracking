## Band Tracking System ##

## Proje Açıklaması

**Band Tracking System**, bir video içerisindeki belirli bir **çalışma alanı** içinde bulunan kişilerin anlık ve periyodik sayımını yapar.  
YOLOv8 nesne algılama modeli kullanılarak insan tespiti gerçekleştirilir, ve sonuçlar düzenli olarak JSON formatında kaydedilir.

---
``` band_tracking/ ├── main.py # Tüm akışı organize eden ana dosya ├── config.py # Konfigürasyonları yükler ve CUDA kontrolü yapar ├── logger.py # Logger ayarlarını yapar (gürültüleri bastırır) ├── model_loader.py # YOLO modelini yükler ├── display_handler.py # Görsel işlemler: kutu çizimi, video gösterimi ├── video_handler.py # Video işlemleri ve log klasörü yönetimi ├── processing.py # Video işleme, insan sayma ve loglama hesaplamaları ├── utils.py # Yardımcı fonksiyonlar (zaman işlemleri gibi) ├── config.json # Ayarların bulunduğu dosya ├── requirements.txt # Gerekli kütüphane listesi ├── models/ # YOLO modellerinin yüklü olduğu klasör ├── videos/ # Örnek videoların yüklü olduğu klasör └── logs/ # Çıktı JSON dosyalarının kaydedildiği klasör ```
---

## Dosya Görevleri

### `main.py`
- **Tüm projenin yönetildiği dosyadır.**
- `config.py` ile ayarları yükler.
- `logger.py` ile log seviyesini ayarlar.
- `model_loader.py` ile YOLO modelini yükler.
- `video_handler.py` ile videoyu açar ve video bilgilerini alır.
- `processing.py` içindeki `process_video()` fonksiyonunu çağırarak:
  - **Frame'leri işler, insanları sayar, loglar.**
- Sonuçları ve süre bilgisini terminalde yazdırır.

---

### `config.py`
- **`config.json`** dosyasını okuyarak model yolu, video yolu, çalışma alanı, eşik değerleri gibi bilgileri çeker.
- `is_cuda_available()` fonksiyonu ile cihazın **GPU destekleyip desteklemediğini** kontrol eder.

---

### `logger.py`
- Ultralytics YOLO modellerinde oluşan **gereksiz bilgi mesajlarını** bastırmak için `LOGGER` seviyesini `ERROR` yapar.

---

### `model_loader.py`
- Verilen model yoluna göre **YOLOv8 modelini** yükler.
- Eğer CUDA destekliyorsa, modeli otomatik olarak **GPU'ya taşır**.

---

### `video_handler.py`
- Belirtilen video dosyasını **OpenCV** ile açar.
- Videonun:
  - FPS değeri
  - Toplam kare sayısı
  gibi bilgilerini çeker.
- Günün tarihine göre (`YYYY-MM-DD` formatında) **log klasörü oluşturur**.
  
---

### `utils.py`
- Basit yardımcı işlemleri içerir.
- Örneğin:
  - Video başlangıç saatini almak (`09:00` sabit saati gibi).
  
---

### `processing.py` (Hesaplama Merkezi)

İşlem adımları:
1. **Frame Okuma**: Videodan her saniye bir frame alınır.
   - Eğer `NORMAL_SPEED = True` ise her frame işlenir (normal hızda video).
2. **3 Frame Batch**: Frame'ler 3'lü gruplar halinde batch yapılır (3 saniyelik pencere).
3. **Konsensüs ile Kişi Sayımı**:
   - Ortadaki frame esas alınır, 1. ve 3. frame ile karşılaştırılarak gerçek kişi tespiti yapılır.
4. **Çizim ve Gösterim (Opsiyonel)**:
   - Eğer `DRAW_DETECTIONS = True` ise her frame üzerinde çalışma alanı, kişi kutuları ve merkez noktaları çizilir.
   - Eğer `SHOW_VIDEO = True` ise işlenen frame ekranda gösterilir.
5. **3 Saniyelik Chunk Oluşturma**:
   - Her 3 saniyede bir kişi sayısı ölçülür ve `time_range`, `person_count` bilgileriyle kaydedilir.
6. **15 Saniyelik Chunk Oluşturma**:
   - 5 adet ardışık 3 saniyelik chunk birleştirilir.
   - Arada en fazla bir sapmaya izin verilir.
   - Eğer son küçük chunk farklıysa, `change_detected: true` olarak işaretlenir.
7. **60 Saniyelik Chunk Oluşturma**:
   - 4 adet 15 saniyelik chunk ardışık geldiğinde 60 saniyelik büyük chunk oluşturulur.
   - `change_detected` alanı dikkate alınmaksızın birleştirme yapılır.
8. **Loglama**:
   - 3 saniyelik, 15 saniyelik ve 60 saniyelik chunklar ilgili JSON dosyalarına kaydedilir.

---

## Loglama Yapısı

İşlem sonunda `logs/YYYY-MM-DD/` klasörü altında 3 farklı JSON dosyası oluşur:

| Dosya Adı | Açıklama |
|:---------|:---------|
| `3_sec_chunks.json` | Her 3 saniyede bir ölçülen kişi sayıları. |
| `15_sec_chunks.json` | 15 saniyelik periyotta sabit kişi sayıları ve değişim bilgisi (`change_detected`). |
| `60_sec_chunks.json` | 60 saniye boyunca oluşan stabil dönemler. |

### JSON Alanları:

| Alan | Açıklama |
|:----|:---------|
| `time_range` | Başlangıç ve bitiş zamanı (örneğin `"09:00:00-09:00:03"`) |
| `person_count` | Belirlenen sürede tespit edilen kişi sayısı |
| `change_detected` | Eğer 15 saniyelik chunkın sonunda kişi sayısı değiştiyse `true`, aksi takdirde `false` |

'''
#############################################################################
                                                                            #
*** Kurulum ve Kullanım ***                                                 #
                                                                            #
**Gerekli Kütüphaneleri Yükle**                                             #
pip install -r requirements.txt                                             #
                                                                            #
**config.json Dosyasını Düzenle**                                           #
{                                                                           #
  "model_path": "models/yolov8x.pt",                                        #
  "video_path": "videos/demo4.mp4",                                         #
  "conf_threshold": 0.5,                                                    #
  "iou_threshold": 0.5,                                                     #
  "batch_size": 1,                                                          #
  "working_zone": [[700, 50], [1200, 50], [1200, 710], [700, 710]]          #
}                                                                           #
                                                                            #
**Projeyi Çalıştır**                                                        #
python main.py                                                              #
                                                                            #
#############################################################################
'''