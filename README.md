# Araç Kasası Sınıflandırma ve Nesne Tespiti (YOLOv8)

**Tokat Gaziosmanpaşa Üniversitesi - Bilgisayar Programcılığı**
**Geliştirici:** Emir Ali Seçkiner (eaS)

Bu proje, görüntü ve videolardaki araçları kasa tiplerine göre (Coupe, Hatchback, SUV, Sedan) tespit eden, YOLOv8 tabanlı bir derin öğrenme uygulamasıdır. Proje, model eğitiminin yanı sıra kullanıcı dostu bir arayüz (GUI) ile test ve manuel etiketleme imkanı sunmaktadır.

## 1. Veri Seti Oluşturma ve Ön İşleme
* **Sınıflar:** Model; `Coupe`, `Hatchback`, `SUV` ve `Sedan` olmak üzere 4 farklı araç kasa tipini tanımaktadır.
* **Veri Toplama ve Çeşitlilik:** Veri seti tamamen bu proje için özel olarak toplanmış ve yaklaşık 1000 adet görüntüden oluşturulmuştur. Modelin gerçek hayat senaryolarında (farklı açılar, değişen ışık koşulları, karmaşık arka planlar ve farklı ölçekler) başarılı olması için veri çeşitliliğine maksimum özen gösterilmiştir.
* **Etiketleme (Bounding Box):** Görüntülerdeki araçlar tek tek manuel olarak, yüksek doğrulukla ve tutarlı sınırlayıcı kutular (bounding box) ile etiketlenmiştir. Bu işlem için projenin kendi içinde geliştirilen "Manuel Etiketleme Arayüzü" kullanılmıştır.
* **Ön İşleme & Ayrım:** Veri seti Train, Validation ve Test olmak üzere üç alt kümeye ayrılmıştır. Eğitim öncesi eksik veya hatalı görüntüler temizlenmiş ve modelin genelleme yeteneğini artırmak için veri artırma (augmentation) teknikleri uygulanmıştır.

## 2. Model Seçimi ve Mimari
* Bu projede hız ve doğruluk dengesi sebebiyle son teknoloji nesne tespiti modellerinden **YOLOv8** tercih edilmiştir. 
* Model sıfırdan eğitilmemiş, önceden eğitilmiş ağırlıklar kullanılarak **Transfer Learning** (Transfer Öğrenme) yöntemiyle kendi özel veri setimize uyarlanmıştır.

## 3. Eğitim (Training) ve Performans
Eğitim sürecine ait tüm kayıp (loss) metrikleri, mAP, Precision ve Recall grafikleri `runs/detect/araba_kasasi_v1/` dizininde bulunmaktadır. Parametreler (epoch, lr vb.) veri setinin boyutuna göre optimize edilmiştir.
* **Sonuç Analizi:** Tanımlama ve test sonuçları yüzdelik dilime göre %85'in üzerinde çıkmış olup. Başarı ile Araç Kasaları Tanımlanmıştır. (Coupe,SUV,Sedan ve Hatchback)

## 4. Kullanım ve Çalışır Sistem (Arayüz Özellikleri)
Proje, test sürecini kolaylaştırmak adına Tkinter kütüphanesi ile tamamen özgün olarak geliştirilmiş bir arayüze sahiptir.

1. **Gerekli Kütüphanelerin Kurulumu:**
   ```bash
   pip install -r requirements.txt
Uygulamayı Başlatma:

Bash
python main.py
Arayüz Modülleri:

Resim Analiz Modu: Seçilen statik bir görüntü üzerinde nesne tespiti yapar ve sonucu ekranda bounding box ile gösterir.

Video Analiz Modu: Seçilen bir video dosyası üzerinde kare kare (frame-by-frame) gerçek zamanlı nesne tespiti ve takibi yapar.

Seri Üretim Etiketleme Modu (Bonus Katkı): Veri setini büyütmek veya yeni veriler eklemek için klavye ve fare kısayolları ile donatılmış özel bir manuel etiketleme aracı geliştirilmiştir. Hızlıca bounding-box çizilip .txt formatında YOLO etiketleri oluşturulabilir.

5. Dosya Yapısı
main.py: Arayüz ve tespit işlemlerinin yapıldığı ana kaynak kodu.

v3.pt: Eğitilmiş en güncel ve optimize model ağırlık dosyası.

data.yaml: Veri seti sınıflarını ve yollarını barındıran yapılandırma dosyası.

runs/: Eğitim sürecine ait performans grafikleri, matrisler ve metrikler.