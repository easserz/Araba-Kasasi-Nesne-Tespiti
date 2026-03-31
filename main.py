import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import shutil
from ultralytics import YOLO

MODEL_DOSYASI = "v3.pt"
model = None

# Sınıflar
SINIFLAR = ['Coupe', 'Hatchback', 'SUV', 'Sedan']

# Klasör oluşturma
if not os.path.exists("dataset/images"): os.makedirs("dataset/images")
if not os.path.exists("dataset/labels"): os.makedirs("dataset/labels")


def modeli_yukle():
    global model
    if not os.path.exists(MODEL_DOSYASI):
        messagebox.showerror("Kritik Hata ❌", f"'{MODEL_DOSYASI}' bulunamadı!\nBu dosya ile kod aynı klasörde olmalı.")
        return False
    try:
        model = YOLO(MODEL_DOSYASI)
        return True
    except Exception as e:
        messagebox.showerror("Model Hatası", f"Model yüklenirken hata oluştu:\n{e}")
        return False


#   VİDEO ANALİZ MODU  
def video_analiz_et():
    global model
    if model is None:
        if not modeli_yukle(): return

    # Video dosyasını seçtirme
    dosya_yolu = filedialog.askopenfilename(
        title="Araba Videosu Seç",
        filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not dosya_yolu: return

    # OpenCV ile videoyu başlat
    cap = cv2.VideoCapture(dosya_yolu)

    if not cap.isOpened():
        messagebox.showerror("Hata", "Video açılamadı!")
        return

    messagebox.showinfo("Bilgi", "Video başlatılıyor...\nÇıkış yapmak için klavyeden 'Q' tuşuna basabilirsiniz.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video bittiğinde döngüden çık

        # GÜNCELLEME 1: conf=0.25'e düşürüldü ve imgsz=640 eklendi
        sonuclar = model.track(source=frame, conf=0.30, imgsz=640, persist=True, tracker="bytetrack.yaml", verbose=False)

        # Etiketlenmiş frame'i al
        cizili_frame = sonuclar[0].plot()

        # GÜNCELLEME 2: Görüntü çok büyükse ekrana sığdırmak ve akıcılığı artırmak için küçült (1280px genişlik)
        frame_h, frame_w = cizili_frame.shape[:2]
        if frame_w > 1280:
            oran = 1280 / frame_w
            cizili_frame = cv2.resize(cizili_frame, (1280, int(frame_h * oran)))

        # Ekranda göster
        cv2.imshow("eaS - Video Analiz Ekrani (Cikmak icin Q'ya basin)", cizili_frame)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Belleği temizle
    cap.release()
    cv2.destroyAllWindows()


#   RESİM ANALİZ MODU  
def analiz_modu_ac():
    analiz_pen = tk.Toplevel()
    analiz_pen.title("Nesne Tanıma")
    analiz_pen.geometry("900x750")
    analiz_pen.configure(bg="#39006e")

    def resim_analiz_et():
        global model
        if model is None:
            if not modeli_yukle(): return

        dosya_yolu = filedialog.askopenfilename(title="Araba Fotoğrafı Seç",
                                                filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.webp")])
        if not dosya_yolu: return

        try:
            lbl_durum.config(text="Durum: Yapay Zeka Analiz Ediyor... ⏳", fg="#ff9d00")
            analiz_pen.update()

            sonuclar = model.predict(source=dosya_yolu, conf=0.5)
            cizili_bgr = sonuclar[0].plot()
            cizili_rgb = cv2.cvtColor(cizili_bgr, cv2.COLOR_BGR2RGB)

            pil_resmi = Image.fromarray(cizili_rgb)
            pil_resmi.thumbnail((800, 500))
            tk_resmi = ImageTk.PhotoImage(pil_resmi)

            lbl_resim.config(image=tk_resmi, text="", width=tk_resmi.width(), height=tk_resmi.height())
            lbl_resim.image = tk_resmi

            lbl_durum.config(text="Durum: Analiz Başarılı! ✅", fg="#00ff6a")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    tk.Label(analiz_pen, text="Araç Kasası Nesne Tanımlama", font=("Segoe UI", 20, "bold"), bg="#39006e",
             fg="#FFFFFF").pack(pady=15)
    tk.Button(analiz_pen, text="Fotoğraf Yükle ve Tara", font=("Segoe UI", 12, "bold"), bg="#5c00b0", fg="white",
              command=resim_analiz_et).pack(pady=10)
    lbl_durum = tk.Label(analiz_pen, text="Durum: Bekleniyor...", bg="#39006e", fg="white", font=("Segoe UI", 12))
    lbl_durum.pack(pady=5)
    lbl_resim = tk.Label(analiz_pen, text="[ Sonuç Burada Görünecek ]", bg="#250047", fg="white", width=70, height=22)
    lbl_resim.pack(pady=15)


#   ETİKETLEME MODU  
class EtiketlemePenceresi:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("Manuel Etiketleme ve Veri Seti Oluşturma (SERİ ÜRETİM MODU)")
        self.top.geometry("1000x800")
        self.top.configure(bg="#eee")

        # Yeni Liste ve Sayaç Değişkenleri
        self.image_list = []
        self.current_index = 0

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.current_img_path = None
        self.pil_image_original = None
        self.pil_image_resized = None
        self.tk_image = None
        self.bbox = None
        self.resize_ratio = 1.0

        # Kontrol Paneli
        panel = tk.Frame(self.top, bg="#ddd", pady=10)
        panel.pack(fill=tk.X)

        tk.Button(panel, text="📂 Klasör Seç", command=self.klasor_sec, bg="#39006e", fg="white",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=10)

        tk.Label(panel, text="Sınıf Seç (1-4):", bg="#ddd", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.sinif_combo = ttk.Combobox(panel, values=SINIFLAR, state="readonly", width=15)
        self.sinif_combo.current(0)
        self.sinif_combo.pack(side=tk.LEFT, padx=5)

        tk.Button(panel, text="💾 KAYDET (Ctrl+S)", command=self.etiketi_kaydet, bg="green", fg="white",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=10)

        tk.Button(panel, text="⏭️ Sonraki Resim (Sağ Tık)", command=self.resmi_atla, bg="orange", fg="black",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.lbl_info = tk.Label(panel, text="Bekleniyor... By eaS", bg="#ddd", font=("Segoe UI", 10, "bold"))
        self.lbl_info.pack(side=tk.RIGHT, padx=10)

        self.canvas = tk.Canvas(self.top, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.cizim_basla)
        self.canvas.bind("<B1-Motion>", self.cizim_yap)
        self.canvas.bind("<ButtonRelease-1>", self.cizim_bitir)

        # --- KLAVYE VE FARE KISAYOLLARI (SPEEDRUN MODU) ---
        self.top.bind("<Control-s>", lambda event: self.etiketi_kaydet())
        self.top.bind("<Control-S>", lambda event: self.etiketi_kaydet())

        self.top.bind("1", lambda event: self.kisayol_sinif_sec(0))
        self.top.bind("2", lambda event: self.kisayol_sinif_sec(1))
        self.top.bind("3", lambda event: self.kisayol_sinif_sec(2))
        self.top.bind("4", lambda event: self.kisayol_sinif_sec(3))

        # SAĞ TIK ile sonraki fotoğrafa geçme (<Button-3> Tkinter'da sağ tık demektir)
        self.top.bind("<Button-3>", lambda event: self.resmi_atla())
        self.canvas.bind("<Button-3>", lambda event: self.resmi_atla())

    def kisayol_sinif_sec(self, index):
        self.sinif_combo.current(index)
        self.lbl_info.config(text=f"Sınıf: {SINIFLAR[index]} seçildi! (Seri Mod)")

    def klasor_sec(self):
        folder_path = filedialog.askdirectory(title="Fotoğrafların Olduğu Klasörü Seç")
        if not folder_path: return

        valid_ext = (".jpg", ".jpeg", ".png")
        self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                           f.lower().endswith(valid_ext)]

        if not self.image_list:
            messagebox.showwarning("Hata", "Seçilen klasörde resim bulunamadı!")
            return

        self.image_list.sort()
        self.current_index = 0
        self.resim_goster()

    def resim_goster(self):
        if self.current_index >= len(self.image_list):
            messagebox.showinfo("Tebrikler!", "Klasördeki tüm fotoğrafları etiketledin, eline sağlık!")
            self.canvas.delete("all")
            self.lbl_info.config(text="Tüm işlemler tamamlandı.")
            return

        filename = self.image_list[self.current_index]
        self.current_img_path = filename
        self.pil_image_original = Image.open(filename)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 100: canvas_width = 800
        if canvas_height < 100: canvas_height = 600

        img_w, img_h = self.pil_image_original.size
        ratio = min(canvas_width / img_w, canvas_height / img_h, 1.0)
        self.resize_ratio = ratio
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        self.pil_image_resized = self.pil_image_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.pil_image_resized)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

        self.bbox = None
        self.rect = None
        self.lbl_info.config(
            text=f"Resim {self.current_index + 1} / {len(self.image_list)} | {os.path.basename(filename)}")

    def cizim_basla(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red",
                                                 width=2)

    def cizim_yap(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def cizim_bitir(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        self.bbox = (x1, y1, x2, y2)

    def resmi_atla(self):
        if self.image_list:
            self.current_index += 1
            self.resim_goster()

    def etiketi_kaydet(self):
        if not self.current_img_path or not self.bbox:
            return

        filename = os.path.basename(self.current_img_path)
        hedef_resim_yolu = os.path.abspath(f"dataset/images/{filename}")
        mevcut_resim_yolu = os.path.abspath(self.current_img_path)

        os.makedirs("dataset/images", exist_ok=True)
        os.makedirs("dataset/labels", exist_ok=True)

        if hedef_resim_yolu != mevcut_resim_yolu:
            shutil.copy(self.current_img_path, hedef_resim_yolu)

        img_w, img_h = self.pil_image_resized.size
        x1, y1, x2, y2 = self.bbox
        x1 = max(0, x1);
        y1 = max(0, y1)
        x2 = min(img_w, x2);
        y2 = min(img_h, y2)

        dw = 1. / img_w
        dh = 1. / img_h
        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + w / 2) * dw
        y_center = (y1 + h / 2) * dh
        width = w * dw
        height = h * dh

        secilen_sinif = self.sinif_combo.get()
        sinif_id = SINIFLAR.index(secilen_sinif)

        txt_name = os.path.splitext(filename)[0] + ".txt"

        with open(f"dataset/labels/{txt_name}", "a") as f:
            f.write(f"{sinif_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        self.canvas.delete(self.rect)
        self.rect = None
        self.bbox = None
        self.lbl_info.config(text=f"Kaydedildi! ({secilen_sinif}) - Sıradaki aracı çizebilirsin.")


#   ANA MENÜ  
ana_pencere = tk.Tk()
ana_pencere.title("Araç Kasası Tanımlama By eaS")
# Buton sığsın diye pencereyi biraz büyüttüm
ana_pencere.geometry("500x550")
ana_pencere.configure(bg="#39006e")

tk.Label(ana_pencere, text="Nesne Tanıma İle\nAraba Kasası Tanıma", font=("Segoe UI", 20, "bold"), bg="#39006e",
         fg="#FFFFFF").pack(pady=30)

# Resim Analizi Butonu
btn_analiz = tk.Button(ana_pencere, text="🖼️ Resim Seç ve Analiz Et", font=("Segoe UI", 12, "bold"), bg="#5c00b0",
                       fg="white", width=25, height=2, cursor="hand2", command=analiz_modu_ac)
btn_analiz.pack(pady=10)

# YENİ EKLENEN VİDEO ANALİZİ BUTONU
btn_video = tk.Button(ana_pencere, text="🎬 Video Seç ve Analiz Et", font=("Segoe UI", 12, "bold"), bg="#008080",
                      fg="white", width=25, height=2, cursor="hand2", command=video_analiz_et)
btn_video.pack(pady=10)

# Etiketleme Butonu
btn_etiket = tk.Button(ana_pencere, text="✏️ Manuel Etiketleme Modu", font=("Segoe UI", 12, "bold"), bg="#e05a00",
                       fg="white", width=25, height=2, cursor="hand2", command=lambda: EtiketlemePenceresi(ana_pencere))
btn_etiket.pack(pady=10)

tk.Label(ana_pencere, text="v1.1 - eaS Dev", font=("Segoe UI", 10), bg="#39006e", fg="#999").pack(side=tk.BOTTOM,
                                                                                                  pady=10)

ana_pencere.mainloop()
