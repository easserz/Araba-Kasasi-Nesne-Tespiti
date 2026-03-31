from ultralytics import YOLO

def model_egit():
    model = YOLO('yolov8m.pt') 
    
    print("Eğitim Başlatılıyor!")
    
    sonuclar = model.train(
        data='data.yaml',       # Tüm sınıfları tutan dosya
        epochs=100,             # Verisetini 100 Kez Eğitecek.
        imgsz=640,              # Netlik için 640x640 çözünürlük
        batch=16,               # Tek seferde alınacak fotoğraf sayısı
        patience=20,            # Verisetini gereksiz tekrarlayarak öğretmeyi önler.
        name='araba_kasasi_v1'  # Sonuçların kaydedileceği klasör
    )
    
    print("Eğitim bitti By eaS")

if __name__ == '__main__':
    model_egit()